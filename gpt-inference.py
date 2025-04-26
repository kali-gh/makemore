from gpt import BigramLanguageModel, decode
import torch
model = BigramLanguageModel()

model.to('cuda')
state_dict = torch.load('model.pt')


model.load_state_dict(state_dict)
model.eval()


from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(model)


idx = torch.zeros((1,1), dtype=torch.long, device='cuda')
with open('out.txt', 'w') as f:
    out = decode(model.generate(idx, max_new_tokens=300)[0].tolist())
    f.write(out)

