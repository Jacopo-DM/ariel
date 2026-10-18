"""Issue unique node and innovation IDs, optionally persisted to disk."""

import json
from pathlib import Path


class IdManager:
    """
    Manages and persists unique innovation and node IDs for a NEAT run.

    Supports two modes:
    1. Persistent Mode (default): Loads and saves its state from a JSON file.
    2. In-Memory Mode: Initializes with specified starting IDs, useful for
      tests.
    """

    def __init__(
        self,
        save_path: str | Path = "id_state.json",
        node_start: int | None = None,
        innov_start: int | None = None,
    ) -> None:
        """
        Initialize the IdManager.

        If node_start and innov_start are provided, the manager starts with
        those values in-memory. Otherwise, it attempts to load its state
        from the save_path.

        Args:
            save_path (Union[str, Path]): Path for saving/loading the ID state.
            node_start (int, optional): The last used ID for nodes. Defaults to
                None.
            innov_start (int, optional): The last used ID for innovations.
                Defaults to None.
        """
        self.save_path = Path(save_path)

        if node_start is not None and innov_start is not None:
            # In-Memory Mode: Use the provided starting values directly.
            self._node_id = node_start
            self._innov_id = innov_start
        else:
            # Persistent Mode: Load from file or start fresh.
            self._node_id = 0
            self._innov_id = 0
            self.load()

    def get_next_node_id(self) -> int:
        """Return the next available node ID and increments the counter.

        Returns
        -------
        int
            The next free node id.
        """
        self._node_id += 1
        return self._node_id

    def get_next_innov_id(self) -> int:
        """Return the next innovation ID and increment the counter.

        Returns
        -------
        int
            The next free innovation id.
        """
        self._innov_id += 1
        return self._innov_id

    def set_initial_ids(self, num_inputs: int, num_outputs: int) -> None:
        """
        Set the initial counters based on a genome's topology.

        Call this once at the start of a new persistent run.
        """
        # Node IDs are 0-indexed. The last used ID is total_nodes - 1.
        self._node_id = num_inputs + num_outputs - 1

        # Innovation IDs start after initial connections. Last used ID is
        # total_conns - 1.
        self._innov_id = (num_inputs * num_outputs) - 1
        print(  # ruff: ignore[print]
            f"IdManager initialized: Next Node ID starts at "
            f"{self._node_id + 1}, Next Innov ID starts at "
            f"{self._innov_id + 1}."
        )

    def save(self) -> None:
        """Save the current ID counters to the specified JSON file."""
        state = {"next_node_id": self._node_id, "next_innov_id": self._innov_id}
        try:
            with self.save_path.open("w") as f:
                json.dump(state, f, indent=4)
        except OSError as e:
            print(f"Error saving ID state: {e}")  # ruff: ignore[print]

    def load(self) -> None:
        """Load the ID counters from the specified JSON file if it exists."""
        if not self.save_path.exists():
            return

        try:
            with self.save_path.open("r") as f:
                state = json.load(f)
                self._node_id = state.get("next_node_id", 0)
                self._innov_id = state.get("next_innov_id", 0)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading ID state: {e}. Starting with fresh IDs.")  # ruff: ignore[print]
            self._node_id = 0
            self._innov_id = 0
