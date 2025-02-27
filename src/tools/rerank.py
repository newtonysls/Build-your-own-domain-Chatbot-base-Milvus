import torch


def rerank(model, tokenizer, pairs):
    with torch.no_grad():
        model.eval()
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(model.device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        indices = scores.sort(descending=True).indices.tolist()
        return scores, indices