import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def tensor_to_image(tensor):
	unloader = transforms.ToPILImage()
	image = tensor.clone()
	image = image.squeeze(0)
	image = unloader(image)
	return image

def my_hook(module, input, output):
	outputs.append(output)

def content_loss(content_output, generated_output):
	_, channels, width, height = content_output.shape
	loss = .5 * torch.sum(torch.square(torch.subtract(content_output, generated_output)))
	return loss

def style_loss(style_outputs, generated_outputs):
	loss = 0
	for i in range(len(style_outputs)):
		style_output, generated_output = style_outputs[i], generated_outputs[i]
		_, channels, width, height = style_output.shape
		GS = gram(style_output)
		GG = gram(generated_output)
		loss += torch.sum(torch.square(torch.subtract(GS, GG))) / (4 * channels**2 * width**2 * height**2)

	loss /= len(style_outputs)
	return loss 

def gram(output):
	_, channels, width, height = output.shape
	output = output.view(channels, -1)
	answer = torch.matmul(output, output.t())
	return answer

def total_variation_loss(image):
	x_diffs = torch.sub(image[:, :, 1:, :], image[:, :, :-1, :])
	y_diffs = torch.sub(image[:, :, :, 1:], image[:, :, :, :-1])

	return .5 * (torch.abs(x_diffs).mean() + torch.abs(y_diffs).mean())


n_epochs = 10000
use_avgpool = True
content_path = "images/Louvre.jpg"
style_path = "images/Starry Night.jpg"
content_layer = [21]
style_layers = [0, 5, 10, 19, 28]
c_coeff, s_coeff, t_v_coeff = 1, 10**4, 100

vgg19 = models.vgg19(weights='VGG19_Weights.DEFAULT')
vgg19 = vgg19.features[:style_layers[-1] + 1]

if use_avgpool:
	for name, module in vgg19.named_modules(): # replacing max pooling with average pooling
		if isinstance(module, nn.MaxPool2d): # as suggested in the original 2015 paper
			setattr(vgg19, name, nn.AvgPool2d(module.kernel_size, module.stride, module.padding))

transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()
])

c_img = Image.open(content_path)
content_img = transform(c_img).unsqueeze(0)
s_img = Image.open(style_path)
style_img = transform(s_img).unsqueeze(0)

tracked_layers = [0, 5, 10, 19, 25, 28]
content_index = 4
style_indices = [0, 1, 2, 3, 5]

outputs = []
for layer in tracked_layers:
	vgg19[layer].register_forward_hook(my_hook)

c = vgg19(content_img)
content_output = outputs[content_index]
sc_outputs = [outputs[i] for i in style_indices]
outputs = []
s = vgg19(style_img)
style_outputs = [outputs[i] for i in style_indices]
cs_outputs = outputs[content_index]
print (content_loss(content_output, cs_outputs))
print (style_loss(style_outputs, sc_outputs))
outputs = []
generated_img = content_img.clone()
generated_img.requires_grad_(True)

optimizer = torch.optim.Adam([generated_img], lr=0.01)

for epoch in range(n_epochs):
	g = vgg19(generated_img)
	c_loss = content_loss(content_output, outputs[content_index])
	s_loss = style_loss(style_outputs, [outputs[i] for i in style_indices])
	t_v_loss = total_variation_loss(generated_img)
	t_loss = c_coeff * c_loss + s_coeff * s_loss + t_v_coeff * t_v_loss
	outputs = []
	if epoch > 0 and epoch % 100 == 0:
		image = tensor_to_image(generated_img)
		plt.title('Generated image at epoch ' + str(epoch))
		save_path = 'images/generated/generated ' + str(epoch) + '.png'
		plt.imsave(save_path, image / np.max(image))
		print (epoch, t_loss, c_loss, s_loss, t_v_loss)
		if epoch % 25000 == 0:
			plt.imshow(image)
			plt.show()


	optimizer.zero_grad()
	t_loss.backward(retain_graph=True)
	optimizer.step()


