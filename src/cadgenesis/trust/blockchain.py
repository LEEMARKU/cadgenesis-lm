"""Blockchain adapter for optional immutable trust layer."""

from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any


class BlockchainBackend(str, Enum):
    """Supported blockchain backends."""

    LOCAL = "local"
    ETHEREUM = "ethereum"
    HYPERLEDGER = "hyperledger"
    POLYGON = "polygon"
    PRIVATE = "private"


@dataclass
class BlockchainConfig:
    """Configuration for blockchain adapter."""

    backend: BlockchainBackend = BlockchainBackend.LOCAL
    network: str = "mainnet"
    rpc_url: str = ""
    contract_address: str = ""
    private_key: str = ""
    gas_limit: int = 3000000
    gas_price_gwei: int = 20
    confirmations: int = 1
    timeout_seconds: int = 120
    # Hyperledger specific
    channel_name: str = "cadgenesis"
    chaincode_name: str = "trust"
    # Local ledger specific
    local_db_path: str = "./local_ledger.db"


class BlockchainTransaction:
    """Represents a blockchain transaction."""

    def __init__(
        self,
        tx_hash: str,
        block_number: int,
        timestamp: float,
        data: dict[str, Any],
        status: str = "pending",
    ):
        self.tx_hash = tx_hash
        self.block_number = block_number
        self.timestamp = timestamp
        self.data = data
        self.status = status


class BlockchainAdapter(abc.ABC):
    """Abstract base class for blockchain adapters."""

    def __init__(self, config: BlockchainConfig):
        self.config = config
        self._lock = RLock()

    @abc.abstractmethod
    def connect(self) -> bool:
        """Connect to the blockchain network."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the blockchain network."""
        pass

    @abc.abstractmethod
    def submit_record(
        self, record_type: str, payload: dict[str, Any]
    ) -> BlockchainTransaction | None:
        """Submit a record to the blockchain."""
        pass

    @abc.abstractmethod
    def verify_record(self, tx_hash: str) -> tuple[bool, dict[str, Any] | None]:
        """Verify a record exists on the blockchain."""
        pass

    @abc.abstractmethod
    def get_record(self, tx_hash: str) -> dict[str, Any] | None:
        """Retrieve a record from the blockchain."""
        pass

    @abc.abstractmethod
    def get_latest_block(self) -> int:
        """Get the latest block number."""
        pass


class LocalLedgerAdapter(BlockchainAdapter):
    """Local SQLite-based ledger for development/testing (no external blockchain)."""

    def __init__(self, config: BlockchainConfig):
        super().__init__(config)
        import sqlite3

        self._conn = sqlite3.connect(config.local_db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                tx_hash TEXT PRIMARY KEY,
                block_number INTEGER,
                timestamp REAL,
                record_type TEXT,
                payload TEXT,
                status TEXT DEFAULT 'confirmed'
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS blocks (
                block_number INTEGER PRIMARY KEY,
                timestamp REAL,
                prev_hash TEXT,
                merkle_root TEXT
            )
            """
        )
        self._conn.commit()

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        self._conn.close()

    def submit_record(
        self, record_type: str, payload: dict[str, Any]
    ) -> BlockchainTransaction | None:
        import hashlib

        tx_hash = hashlib.sha256(
            f"{record_type}{json.dumps(payload, sort_keys=True)}{time.time()}".encode()
        ).hexdigest()
        block_number = self.get_latest_block() + 1
        timestamp = time.time()

        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO records (tx_hash, block_number, timestamp, record_type, "
            "payload, status) VALUES (?, ?, ?, ?, ?, ?)",
            (tx_hash, block_number, timestamp, record_type, json.dumps(payload), "confirmed"),
        )
        # Create block
        prev_hash = "0" * 64
        if block_number > 1:
            cursor.execute(
                "SELECT merkle_root FROM blocks WHERE block_number = ?", (block_number - 1,)
            )
            row = cursor.fetchone()
            if row:
                prev_hash = row[0]

        merkle_root = hashlib.sha256(f"{tx_hash}{prev_hash}".encode()).hexdigest()
        cursor.execute(
            "INSERT INTO blocks (block_number, timestamp, prev_hash, merkle_root) "
            "VALUES (?, ?, ?, ?)",
            (block_number, timestamp, prev_hash, merkle_root),
        )
        self._conn.commit()

        return BlockchainTransaction(tx_hash, block_number, timestamp, payload, "confirmed")

    def verify_record(self, tx_hash: str) -> tuple[bool, dict[str, Any] | None]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT payload, status FROM records WHERE tx_hash = ?", (tx_hash,))
        row = cursor.fetchone()
        if row and row[1] == "confirmed":
            return True, json.loads(row[0])
        return False, None

    def get_record(self, tx_hash: str) -> dict[str, Any] | None:
        cursor = self._conn.cursor()
        cursor.execute("SELECT payload FROM records WHERE tx_hash = ?", (tx_hash,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def get_latest_block(self) -> int:
        cursor = self._conn.cursor()
        cursor.execute("SELECT MAX(block_number) FROM blocks")
        row = cursor.fetchone()
        return row[0] if row and row[0] else 0


class EthereumAdapter(BlockchainAdapter):
    """Ethereum/Polygon blockchain adapter using web3.py."""

    def __init__(self, config: BlockchainConfig):
        super().__init__(config)
        self._web3: Any = None
        self._account = None
        self._contract = None

    def connect(self) -> bool:
        try:
            from web3 import Web3
            from web3.middleware import geth_poa_middleware

            self._web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            if "poa" in self.config.network.lower():
                self._web3.middleware_onion.inject(geth_poa_middleware, layer=0)

            if not self._web3.is_connected():
                return False

            if self.config.private_key:
                self._account = self._web3.eth.account.from_key(self.config.private_key)

            # Load contract ABI (simplified - would need actual contract)
            if self.config.contract_address:
                # self._contract = self._web3.eth.contract(
                #     address=self.config.contract_address, abi=CONTRACT_ABI
                # )
                pass

            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        self._web3 = None
        self._account = None
        self._contract = None

    def submit_record(
        self, record_type: str, payload: dict[str, Any]
    ) -> BlockchainTransaction | None:
        if not self._web3 or not self._account:
            return None

        # This is a simplified implementation
        # In production, you'd call the smart contract
        tx_hash = "0x" + "0" * 64  # Placeholder
        return BlockchainTransaction(tx_hash, 0, time.time(), payload, "pending")

    def verify_record(self, tx_hash: str) -> tuple[bool, dict[str, Any] | None]:
        # Would verify on-chain
        return False, None

    def get_record(self, tx_hash: str) -> dict[str, Any] | None:
        # Would retrieve from chain
        return None

    def get_latest_block(self) -> int:
        if self._web3:
            return self._web3.eth.block_number
        return 0


class HyperledgerAdapter(BlockchainAdapter):
    """Hyperledger Fabric adapter."""

    def __init__(self, config: BlockchainConfig):
        super().__init__(config)
        self._gateway = None
        self._network = None
        self._contract = None

    def connect(self) -> bool:
        try:
            # from hyperledger_fabric_gateway import Gateway
            # self._gateway = Gateway(...)
            # self._network = self._gateway.get_network(self.config.channel_name)
            # self._contract = self._network.get_contract(self.config.chaincode_name)
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        if self._gateway:
            self._gateway.close()
        self._gateway = None
        self._network = None
        self._contract = None

    def submit_record(
        self, record_type: str, payload: dict[str, Any]
    ) -> BlockchainTransaction | None:
        # Submit to chaincode
        return None

    def verify_record(self, tx_hash: str) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def get_record(self, tx_hash: str) -> dict[str, Any] | None:
        return None

    def get_latest_block(self) -> int:
        return 0


def create_blockchain_adapter(config: BlockchainConfig) -> BlockchainAdapter:
    """Factory function to create the appropriate blockchain adapter."""
    if config.backend == BlockchainBackend.LOCAL:
        return LocalLedgerAdapter(config)
    elif config.backend in (BlockchainBackend.ETHEREUM, BlockchainBackend.POLYGON):
        return EthereumAdapter(config)
    elif config.backend == BlockchainBackend.HYPERLEDGER:
        return HyperledgerAdapter(config)
    elif config.backend == BlockchainBackend.PRIVATE:
        return LocalLedgerAdapter(config)  # Use local for private chains
    else:
        raise ValueError(f"Unsupported blockchain backend: {config.backend}")
