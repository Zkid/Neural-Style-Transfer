import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def my_hook(module, input, output):
	outputs.append(output)

def content_cost(content_output, generated_output):
	_, channels, width, height = content_output.shape
	loss = .5 * torch.sum(torch.square(torch.subtract(content_output, generated_output)))
	return loss

def style_cost(style_outputs, generated_outputs):
	loss = 0
	for i in range(len(style_outputs)):
		style_output, generated_output = style_outputs[i], generated_outputs[i]
		_, channels, width, height = style_output.shape
		GS = gram(style_output)
		GG = gram(generated_output)
		loss += torch.sum(torch.square(torch.subtract(GS, GG)))

	loss /= (len(style_outputs) * 4 * channels**2 * width**2 * height**2)
	return loss 

def gram(output):
	_, channels, width, height = output.shape
	output = output.view(channels, -1)
	answer = torch.matmul(output, output.t())
	return answer

n_epochs = 10000
use_avgpool = True
content_path = "images/Louvre.jpg"
style_path = "images/Starry Night.jpg"
content_layer = [21]
style_layers = [0, 5, 10, 19, 28]
c_coeff, s_coeff = 1, 4

vgg19 = models.vgg19(weights='VGG19_Weights.DEFAULT')
vgg19_features = vgg19.features[:style_layers[-1] + 1]

if use_avgpool:
	for name, module in vgg19_features.named_modules(): # replacing max pooling with average pooling
		if isinstance(module, nn.MaxPool2d): # as suggested in the original 2015 paper
			setattr(vgg19_features, name, nn.AvgPool2d(module.kernel_size, module.stride, module.padding))

transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()
])

img = Image.open(content_path)
content_img = transform(img).unsqueeze(0)
img = Image.open(style_path)
style_img = transform(img).unsqueeze(0)

tracked_layers = [0, 5, 10, 19, 21, 28]
content_index = 4
style_indices = [0, 1, 2, 3, 5]

outputs = []
for layer in tracked_layers:
	vgg19_features[layer].register_forward_hook(my_hook)

c = vgg19(content_img)
content_output = outputs[content_index]
outputs = []
s = vgg19(style_img)
style_outputs = [outputs[i] for i in style_indices]
print (content_cost(content_output, outputs[4]))
outputs = []
generated_img = content_img.clone().requires_grad_(True)

optimizer = torch.optim.Adam([generated_img], lr=0.01)

for epoch in n_epochs:
	g = vgg19_features(generated_img)
	c_cost = content_cost(content_output, outputs[content_index])
	s_cost = style_cost(style_outputs, outputs[i] for i in style_indices)
	t_cost = c_coeff * c_cost + s_coeff * s_cost

	optimizer.zero_grad()
	t_cost.backward(retain_graph=True)
	optimizer.step()



