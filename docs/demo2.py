from afsl import ActiveDataLoader

context_loader = ActiveDataLoader.initialize(data, target, batch_size=5)
context = data[context_loader.next(model)]
model.add_to_context(context)
