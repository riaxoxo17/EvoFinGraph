# Elliptic Bitcoin Dataset

## Dataset Statistics

- Transactions: 203,769
- Time Steps: 49
- Features per Transaction: 166
- Graph Edges: 234,355

## Nodes

Each node represents a Bitcoin transaction.

## Edges

Each directed edge represents a flow of Bitcoin from one transaction to another.

## Labels

- 0 → Licit
- 1 → Illicit

Unknown labels are removed during preprocessing.

## Temporal Information

The dataset is divided into 49 temporal snapshots.

Each snapshot corresponds to a fixed time interval.

## Future Engineered Features

- Community ID
- Community Size
- Community Density
- Community Growth Rate
- Adaptive Fraud Evolution Index (AFEI)
- Community Risk Score