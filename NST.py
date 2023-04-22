import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
	unloader = transforms.ToPILImage()
	image = tensor.clone()
	image = image.squeeze(0)
	image = unloader(image)
	return image

def content_cost(content_output, generated_output):
	_, n_C, n_H, n_W = content_output.shape
	total = torch.sum(torch.square(torch.subtract(content_output, generated_output)))
	cost = total / (4 * n_C * n_H * n_W)

	return cost

def style_cost(style_output, generated_output):
	cost = 0
	for i in range(len(style_output)):
		so_i = style_output[i]
		go_i = generated_output[i]
		_, n_C, n_H, n_W = so_i.shape
		so_i = so_i.reshape(n_C, -1)
		go_i = go_i.reshape(n_C, -1)

		GC = gram_matrix(so_i)
		GG = gram_matrix(go_i)
		total = torch.sum(torch.square(torch.subtract(GC, GG)))
		cost += total / (4 * n_C**2 * n_H**2 * n_W**2)

	cost /= len(style_output)

	return cost

def gram_matrix(channels):
	return torch.matmul(channels, torch.transpose(channels, 0, 1))

def total_cost(content_cost, style_cost, c_C=10, c_S=40):
	return c_C * content_cost + c_S * style_cost

def get_outputs(name):
    def hook(model, input, output):
        outputs[name] = output
    return hook

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

vgg19 = torchvision.models.vgg19(weights='VGG19_Weights.DEFAULT')
vgg19 = vgg19.features[:29]

transform = transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(224),
	transforms.ToTensor()])
''',
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225])'''

content_image = Image.open("images/Louvre.jpg")
content_tensor = transform(content_image).unsqueeze(0)
style_image = Image.open("images/Monet Poppies.jpg")
style_tensor = transform(style_image).unsqueeze(0)

outputs = {}
style_coeffs = [0, 5, 10, 19, 28]
for style_coeff in style_coeffs:
	vgg19[style_coeff].register_forward_hook(get_outputs('features' + str(style_coeff)))
y = vgg19(content_tensor)
content_output = outputs['features19']
y = vgg19(style_tensor)
style_output = [outputs['features' + str(style_coeffs[i])] for i in range(len(style_coeffs))]
generated_tensor = content_tensor.clone() #+ .1 * torch.randn(content_tensor.shape)
generated_tensor.requires_grad_(True)

n_epochs = 100000
optimizer = torch.optim.Adam([generated_tensor], lr = 0.001)
min_cost = torch.tensor(10**10)
best_tensor = None
for epoch in range(n_epochs):
	y = vgg19(generated_tensor)
	generated_output = [outputs['features' + str(style_coeffs[i])] for i in range(len(style_coeffs))]
	c_cost = content_cost(content_output, generated_output[3])
	s_cost = style_cost(style_output, generated_output)
	t_cost = total_cost(c_cost, s_cost, 10, 40)
	if epoch % 100 == 0:
		print (epoch, t_cost)
		print (c_cost, s_cost)
		print('Memory Usage:')
		print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
		print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
		if epoch >= 1000:
			plt.imshow(tensor_to_image(best_tensor))
			print (min_cost)
			plt.show()
	if t_cost < min_cost:
		min_cost = t_cost
		best_tensor = generated_tensor
	optimizer.zero_grad()
	t_cost.backward(retain_graph=True)
	optimizer.step()

plt.imshow(tensor_to_image(best_tensor))
print (min_cost)
plt.show()
plt.imshow(tensor_to_image(generated_tensor))
print (t_cost)
plt.show()


