# В одной команде работали:
# Пырков Д. А.
# Кочегин В. В.
# Пичаев И. А.
# Чупраков С. В.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='./data/train',
                                                     transform=data_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root='./data/test',
                                                 transform=data_transforms)

    class_names = train_dataset.classes

    batch_size = 10
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=True,  num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                        shuffle=False, num_workers=0) 

    net = torchvision.models.alexnet(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False

    num_classes = 3
    new_classifier = net.classifier[:-1]
    new_classifier.add_module('fc', nn.Linear(4096, num_classes))
    net.classifier = new_classifier
    net = net.to(device)

    num_epochs = 3
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    import time
    t = time.time()
    save_loss = []
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = lossFn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            save_loss.append(loss.item())
            if i%100 == 0:
                print(f'Эпоха {epoch} из {num_epochs} Шаг {i} Ошибка: {loss.item()}')

    print(f'Время обучения: {time.time() - t}')

    # Визуализация потерь
    plt.figure()
    plt.plot(save_loss)
    plt.show()

    # Оценка точности
    correct_predictions = 0
    num_test_samples = len(test_dataset)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)
            _, pred_class = torch.max(pred.data, 1)
            correct_predictions += (pred_class == labels).sum().item()

    print(f'Точность предобученной модели: {100 * correct_predictions / num_test_samples}%')

    # Визуализация примеров
    class_samples = {i: [] for i in range(num_classes)}
    
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            if len(class_samples[label.item()]) < 5:
                class_samples[label.item()].append(img)
        
        if all(len(v) >= 5 for v in class_samples.values()):
            break
    
    plt.figure(figsize=(15, 9))
    for class_idx in range(num_classes):
        for i in range(5):
            plt.subplot(num_classes, 5, class_idx * 5 + i + 1)
            img = class_samples[class_idx][i].numpy().transpose((1, 2, 0))
            
            # Денормализация
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            plt.title(f"Class: {class_names[class_idx]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()