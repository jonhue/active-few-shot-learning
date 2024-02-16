from afsl import ActiveDataLoader, GradientEmebeddings

emb = GradientEmebeddings()
train_loader = ActiveDataLoader(emb, dataset, task, batch_size=32)

while not converged:
    data = train_loader.next(model)
    model.step(data)
