


from torch.utils.data import Dataset



class TraceDataset(Dataset):
    def __init__(self, traces):
        self.traces = traces

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]

        token_features = torch.tensor(
            trace["token_features"], dtype=torch.long
        )  # (T, K)

        access_features = torch.tensor(
            trace["access_features"], dtype=torch.float32
        )  # (T, A)

        label = torch.tensor(trace["label"], dtype=torch.long)

        return token_features, access_features, label


from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

model = CombinedLSTMModel(
    token_vocab_size=len(token_to_id),
    token_embedding_dim=E,
    access_feature_size=A,
    hidden_dim=128,
    output_dim=NUM_CLASSES,
    num_layers=2,
    dropout=0.3,
)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()

    for token_feats, access_feats, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(token_feats, access_feats)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for token_feats, access_feats, labels in val_loader:
        outputs = model(token_feats, access_feats)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Accuracy:", correct / total)


