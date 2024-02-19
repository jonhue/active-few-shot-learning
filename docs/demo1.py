from afsl import ActiveDataLoader

train_loader = ActiveDataLoader.initialize(data, target, batch_size=32)

while not converged:
    batch = data[train_loader.next(model)]
    model.step(batch)
