import math

x = [1, 0, 1]

w_input_hidden = [
    [0.5, -0.2, 0.3],
    [0.4,  0.1, -0.5]
]

b_hidden = [0.1, -0.1]

w_hidden_output = [0.7, 0.2]

b_output = 0.05

def sigmoid(value):
    return 1 / (1 + math.exp(-value))

hidden_net = []
hidden_act = []

for h in range(len(w_input_hidden)):
    net = 0
    for i in range(len(x)):
        net += x[i] * w_input_hidden[h][i]
    net += b_hidden[h]
    hidden_net.append(net)
    hidden_act.append(sigmoid(net))

output_net = 0
for h in range(len(hidden_act)):
    output_net += hidden_act[h] * w_hidden_output[h]

output_net += b_output
output = sigmoid(output_net)

print("Hidden Layer Net Values :", hidden_net)
print("Hidden Layer Activations:", hidden_act)
print("Spam Probability       :", output)
