from afsl import ActiveDataLoader

train_loader = ActiveDataLoader.initialize(dataset, target, batch_size=32)

while not converged:
    batch = dataset[train_loader.next(model)]
    model.step(batch)
