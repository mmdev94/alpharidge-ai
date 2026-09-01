# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import asyncio
import threading
import argparse
import traceback

import bittensor as bt

from alpharidge_ai.base.neuron import BaseNeuron
from alpharidge_ai.utils.config import add_miner_args

from typing import Union


# When deregistered: long sleep between chain probes (minimal CPU / RPC).
DEREGISTERED_IDLE_POLL_S = 600.0


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )
        # The axon handles request processing, allowing validators to send this miner requests.
        # self.axon = bt.axon(
        #     wallet=self.wallet,
        #     config=self.config() if callable(self.config) else self.config,
        # )

        self.axon = bt.Axon(
            wallet=self.wallet,
            config=self.config,
            port=self.config.axon.port,
            ip=self.config.axon.ip,
            external_ip=self.config.axon.external_ip,
            external_port=self.config.axon.external_port,
            max_workers=self.config.axon.max_workers,
        )

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        # Attach forward_score handler for Score synapses
        bt.logging.info(f"Attaching forward_score function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_score
        )
        # Attach forward_validation_result handler for ValidationResult synapses
        bt.logging.info(f"Attaching forward_validation_result function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_validation_result
        )
        # Attach forward_is_alive handler for IsAlive synapses
        bt.logging.info(f"Attaching forward_is_alive function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_is_alive
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()
        self._axon_started: bool = False
        self._idle_notice_sent: bool = False

    def _on_subnet_deregistered(self) -> None:
        """Hook when hotkey leaves the metagraph (subclasses may log once)."""
        if not self._idle_notice_sent:
            bt.logging.warning(
                f"Hotkey deregistered on netuid {self.config.netuid}; "
                f"axon stopped — idle until re-registered."
            )
            self._idle_notice_sent = True

    def _on_subnet_registered(self) -> None:
        """Hook when hotkey returns to the metagraph."""
        self._idle_notice_sent = False
        bt.logging.info(
            f"Hotkey registered on netuid {self.config.netuid} uid={self.uid}; resuming axon."
        )

    def _start_axon(self) -> None:
        if self._axon_started:
            return
        self.sync()
        if not self.is_subnet_registered or self.uid is None:
            return
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        self._axon_started = True
        bt.logging.info(f"Miner starting at block: {self.block}")

    def _stop_axon(self) -> None:
        if not self._axon_started:
            return
        try:
            self.axon.stop()
        except Exception as e:
            bt.logging.debug(f"axon stop: {e}")
        self._axon_started = False

    def _idle_until_registered(self) -> None:
        """Block with minimal activity while deregistered."""
        self._stop_axon()
        self._on_subnet_deregistered()
        poll = float(getattr(self.config.neuron, "deregistered_poll", DEREGISTERED_IDLE_POLL_S))
        while not self.should_exit:
            if self.refresh_registration():
                self._on_subnet_registered()
                return
            time.sleep(max(30.0, poll))

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        try:
            while not self.should_exit:
                if not self.is_subnet_registered:
                    self._idle_until_registered()
                    if self.should_exit:
                        break

                self._start_axon()
                if not self._axon_started:
                    self._idle_until_registered()
                    continue

                while not self.should_exit and self.is_subnet_registered:
                    while (
                        self.block - self.metagraph.last_update[self.uid]
                        < self.config.neuron.epoch_length
                    ):
                        time.sleep(1)
                        if self.should_exit:
                            break
                    if self.should_exit:
                        break

                    self.sync()
                    if not self.is_subnet_registered:
                        break
                    self.step += 1

        except KeyboardInterrupt:
            self._stop_axon()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        except Exception:
            bt.logging.error(traceback.format_exc())
        finally:
            self._stop_axon()

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
