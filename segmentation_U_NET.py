#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import sys
import random
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import cv2
from scipy.ndimage import binary_closing
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
import triangle
from scipy.interpolate import griddata
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import split

random.seed(1338)

class seg_UNet(nn.Module):
    def __init__(self):
        super(seg_UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate=0.5):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
        )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels)
        )

        # Downsampling
        self.conv1 = conv_block(1, 128)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = conv_block(128, 256)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = conv_block(256, 512)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv4 = conv_block(512, 1024)
        self.pool4 = nn.AvgPool2d(2, 2)

        # Bridge
        self.conv5 = conv_block(1024, 2048)
        self.conv6 = conv_block(2048, 2048)

        # Upsampling
        self.upconv7 = upconv_block(2048, 1024)
        self.conv7 = conv_block(2048, 1024)
        self.dropout7 = nn.Dropout(0.3)
        self.upconv8 = upconv_block(1024, 512)
        self.conv8 = conv_block(1024, 512)
        self.dropout8 = nn.Dropout(0.3)
        self.upconv9 = upconv_block(512, 256)
        self.conv9 = conv_block(512, 256)
        self.dropout9 = nn.Dropout(0.3)
        self.upconv10 = upconv_block(256, 128)
        self.conv10 = conv_block(256, 128)

        self.output = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img1):
        #x = torch.cat((img1, img2), dim=1)
        x = img1
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
        x8 = self.pool4(x7)

        x9 = self.conv5(x8)
        x10 = self.conv6(x9)

        x11 = self.upconv7(x10)
        x12 = torch.cat([x11, x7], dim=1)
        x13 = self.conv7(x12)

        x14 = self.upconv8(x13)
        x15 = torch.cat([x14, x5], dim=1)
        x16 = self.conv8(x15)

        x17 = self.upconv9(x16)
        x18 = torch.cat([x17, x3], dim=1)
        x19 = self.conv9(x18)

        x20 = self.upconv10(x19)
        x21 = torch.cat([x20, x1], dim=1)
        x22 = self.conv10(x21)

        out = self.output(x22)

        return out
#%%
with open('large_dataset_with_predictions_gaussian_2.pkl', 'rb') as f:
    data = pickle.load(f)

class CustomDataset(Dataset):
    def __init__(self,images,column_maps, ground_truths,positions,labels,density_maps,ldesc_maps):
        self.images = images
        self.ground_truths = ground_truths
        self.column_maps = column_maps
        self.positions = positions
        self.labels = labels
        self.density_maps = density_maps
        self.ldesc_maps = ldesc_maps

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        raw_image = self.images[idx]
        ground_truth = self.ground_truths[idx]
        #ground_truth = binary_closing(ground_truth, structure=np.ones((5,5))).astype(int)
        density_map = self.density_maps[idx]
        ldesc_map = self.ldesc_maps[idx]
        column_map = self.column_maps[idx]
        positions = self.positions[idx]
        labels = self.labels[idx]

        normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
        normalized_image = normalized_image[np.newaxis, :, :]
        ground_truth = ground_truth[np.newaxis, :, :]
        column_map = column_map[np.newaxis, :, :]
        density_map = density_map[np.newaxis, :, :]
        ldesc_map = ldesc_map[np.newaxis, :, :]
        image = torch.tensor(normalized_image, dtype=torch.float32)
        column_map = torch.tensor(column_map, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.float32)
        density_map = torch.tensor(density_map, dtype=torch.float32)
        ldesc_map = torch.tensor(ldesc_map, dtype=torch.float32)
        ground_truth = ground_truth * (column_map > 0.5)
        return image, ground_truth,column_map,positions,labels,density_map,ldesc_map

def plot_losses(train_losses, validation_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid()
    plt.show()

def preprocess_image(image_path):
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
    normalized_image = normalized_image[np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
    return image_tensor

def postprocess_output(output_tensor):
    output_numpy = output_tensor.detach().cpu().numpy()
    output_image = np.squeeze(output_numpy)
    return output_image    

def compute_regions(output):
    mask = output > 0.5
    mask = mask.int()

    return mask

def label_points(positions, mask):

    mask = mask.squeeze()
    point_coordinates = positions
    pixel_coordinates = np.floor(point_coordinates).astype(int)
    
    point_labels = []
    for pixel_coord in pixel_coordinates:
        point_labels.append(mask[pixel_coord[0], pixel_coord[1]])
    point_labels = np.array(point_labels)
    return point_labels

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        num = 2. * (output * target).sum(dim=(2,3))
        den = output.sum(dim=(2,3)) + target.sum(dim=(2,3)) + self.eps

        return 1 - num / den

def iou_loss(pred, target):
    smooth = 1.  # Adds a smoothing factor to avoid division by zero

    # Flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Intersection is equivalent to True Positive count
    intersection = (pred * target).sum()

    # IoU formula
    total = (pred + target).sum()
    union = total - intersection 

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU
positions = [df[['x','y']].to_numpy()*128 for df in data['dataframes']]
labels = [df['label'].to_numpy() for df in data['dataframes']]
#%%

def triangle_area(a, b, c):
    return 0.5 * abs(a[0]*b[1] + b[0]*c[1] + c[0]*a[1] - a[1]*b[0] - b[1]*c[0] - c[1]*a[0])

def calculate_density(tri, point, vertices):
    indices = np.where((tri.simplices == point).any(axis=1))[0]
    triangles = tri.simplices[indices]
    total_area = sum(triangle_area(vertices[triangle][0], vertices[triangle][1], vertices[triangle][2]) for triangle in triangles)
    neighbors = np.unique(triangles[triangles != point])
    density = total_area / (len(neighbors) + 1)
    return density

density_maps = []

for pos_index, pos in enumerate(positions):
    print(pos_index)
    tri = Delaunay(pos)
    densities = np.zeros(len(pos))
    for i in range(len(pos)):
        densities[i] = calculate_density(tri, i, pos)
    grid_x, grid_y = np.mgrid[0:128, 0:128]
    density_map = griddata(pos, densities, (grid_x, grid_y), method='cubic', fill_value=0)
    density_map = np.array(density_map.T)
    density_maps.append(density_map)
data['density_maps'] = density_maps 

#%%

def add_midpoints(pos, tri):
    #calculate the midpoint of all edges
    midpoints = []
    for i in range(len(tri.simplices)):
        #get the three points of the triangle
        p1 = pos[tri.simplices[i,0]]
        p2 = pos[tri.simplices[i,1]]
        p3 = pos[tri.simplices[i,2]]
        #calculate the midpoints of the edges
        m1 = (p1+p2)/2
        m2 = (p2+p3)/2
        m3 = (p3+p1)/2
        #add the midpoints to the list
        midpoints.append(m1)
        midpoints.append(m2)
        midpoints.append(m3)
    #convert the list to a numpy array
    midpoints = np.array(midpoints)
    #remove duplicate points
    midpoints = np.unique(midpoints, axis=0)
    #add the midpoints to the list of points
    pos = np.concatenate((pos,midpoints), axis=0)
    #calculate a new triangulation
    tri = Delaunay(pos)
    return pos, tri

from scipy.spatial import ConvexHull
from skimage.draw import polygon
from scipy.ndimage import median_filter
from collections import defaultdict

def voronoi_ridge_neighbors(vor):
    ridge_dict = defaultdict(list)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_dict[tuple(sorted((p1, p2)))].extend([v1, v2])

    region_neighbors = defaultdict(set)
    for pr, vr in ridge_dict.items():
        r1, r2 = [vor.point_region[p] for p in pr]
        if all(v >= 0 for v in vr):  # ridge is finite
            region_neighbors[r1].add(r2)
            region_neighbors[r2].add(r1)
    
    return region_neighbors

def calc_lattice_descriptor_map(idx,positions):
    pos_original = positions[idx].copy()
    pos = positions[idx]
    tri = Delaunay(pos)
    pos, tri = add_midpoints(pos, tri)
    pos, tri = add_midpoints(pos, tri)
    vor = Voronoi(pos)

    neighbors = voronoi_ridge_neighbors(vor)
    sim_vals = []
    positions = []
    for region, neighbor_regions in neighbors.items():
        input_point_index = np.where(vor.point_region == region)[0]
        if len(input_point_index) == 0:
            continue
        input_point = vor.points[input_point_index[0]]
        positions.append(input_point)

        #get the points of the region
        points = vor.vertices[vor.regions[region]]
        if np.any(points > 128) or np.any(points < 0):
            #print('outside of image')
            sim_vals.append(0)
            continue
        #get the voronoi point of the region
        point_index = np.where(vor.point_region == region)[0]
        input_point = vor.points[point_index[0]]
        poly1 = Polygon(points)
        #get the neighboring points of the region
        all_ious = []  # Initialize all_ious here
        for nbor in neighbor_regions:
            #get the points of the neighboring region
            nbor_points = vor.vertices[vor.regions[nbor]]
            poly2 = Polygon(nbor_points)
            if np.any(nbor_points > 128) or np.any(nbor_points < 0) or not poly1.is_valid or not poly2.is_valid:
                #print('outside of image')
                continue
            #get the voronoi point of the neighboring region
            nbor_point_index = np.where(vor.point_region == nbor)[0]
            if len(nbor_point_index) == 0:
                continue
            nbor_input_point = vor.points[nbor_point_index[0]]
            #calculate the distance vector between the two voronoi points
            distance_vec = nbor_input_point - input_point
            #calculate the new points of the neighboring region
            nbor_points = nbor_points - distance_vec
            #calculate intersection over union
            poly2 = Polygon(nbor_points)
            intersection = poly1.intersection(poly2).area
            union = unary_union([poly1, poly2]).area
            iou = intersection / union if union else 0
            #print(iou)
            all_ious.append(iou)

        # Move the following lines out of the inner loop
        if len(all_ious) > 0:
            mean_IoU = np.mean(all_ious)
            #std_dev_IoU = np.std(all_ious)
            #cv_IoU = std_dev_IoU / mean_IoU
            sim_vals.append(mean_IoU)
        else:
            sim_vals.append(0)
    #print(len(sim_vals), len(positions))
    #make a grid interpolating between the positions and their corresponding similarity values
    grid_x, grid_y = np.mgrid[0:128, 0:128]
    lattice_descriptor_map = griddata(positions, sim_vals, (grid_x, grid_y), method='linear', fill_value=0)
    #calculate the convex hull of pos_sim, make a binary mask and multiply to grid_z
    hull = ConvexHull(pos_original)
    hull_points = pos_original[hull.vertices]
    x = hull_points[:, 1]
    y = hull_points[:, 0]
    mask = np.zeros_like(lattice_descriptor_map, dtype=bool)
    rr, cc = polygon(y, x)
    mask[rr, cc] = True
    lattice_descriptor_map = lattice_descriptor_map*mask
    # Define a 3x3 mean filter
    size =5
    # Apply mean filter to smooth the grid
    lattice_descriptor_map = median_filter(lattice_descriptor_map, size)
    dx, dy = np.gradient(lattice_descriptor_map)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    
    return gradient_magnitude.T
#%%
plot_nr = 86
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(data['images'][plot_nr], cmap='gray') # add the origin parameter
plt.title('Image')
plt.axis('off')
lattice_descriptor_map = calc_lattice_descriptor_map(plot_nr, positions)
plt.subplot(1, 3, 3)
plt.imshow(lattice_descriptor_map, cmap='hot',vmin=0,vmax=0.3) # add the origin parameter
plt.scatter(positions[plot_nr][:, 0],positions[plot_nr][:, 1], s=3, c='g')
plt.axis('off')
plt.show()
#%%
lattice_descriptor_maps = []

for pos_index, pos in enumerate(positions):
    print(pos_index)
    lattice_descriptor_map = calc_lattice_descriptor_map(pos_index, positions)
    lattice_descriptor_maps.append(lattice_descriptor_map)
data['lattice_descriptor_maps'] = lattice_descriptor_maps 
#%%
#use pickle to save lattice_descriptor_maps
with open('lattice_descriptor_maps_grad.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(lattice_descriptor_maps, f)
#%%
#use pickle to load lattice_descriptor_maps
lattice_descriptor_maps = []
with open('lattice_descriptor_maps_grad.pkl', 'rb') as f:
    lattice_descriptor_maps = pickle.load(f)
data['lattice_descriptor_maps'] = lattice_descriptor_maps 
#%%
plot_nr =437
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(data['images'][plot_nr], cmap='gray') # add the origin parameter
plt.title('Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(density_maps[plot_nr], cmap='hot') # add the origin parameter
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(data['lattice_descriptor_maps'][plot_nr], cmap='hot') # add the origin parameter
plt.scatter(positions[plot_nr][:, 0],positions[plot_nr][:, 1], s=3, c='g')
plt.axis('off')
plt.show()

#%%

train_images, validation_images, train_ground_truths, validation_ground_truths, train_predictions, validation_predictions, train_positions, validation_positions, train_labels, validation_labels, train_dmaps,validation_dmaps, train_ldescriptor_maps, validation_ldescriptor_maps = train_test_split(
    data['images'], data['segmented_images'],data['predictions'], positions, labels,data['density_maps'],data['lattice_descriptor_maps'], test_size=0.25, random_state=1337)

train_dataset = CustomDataset(train_images,train_predictions, train_ground_truths,train_positions,train_labels,train_dmaps,train_ldescriptor_maps)
validation_dataset = CustomDataset(validation_images,validation_predictions, validation_ground_truths,validation_positions,validation_labels,validation_dmaps,validation_ldescriptor_maps)

def collate_fn(batch):
    images, ground_truths, column_maps, positions, labels, dmaps, ldesc_maps = zip(*batch)
    
    # Stack images, ground_truths, column_maps as usual
    images = torch.stack(images)
    ground_truths = torch.stack(ground_truths)
    column_maps = torch.stack(column_maps)
    dmaps = torch.stack(dmaps)
    ldesc_maps = torch.stack(ldesc_maps)

    # Padding variable-length sequences
    positions = pad_sequence([torch.tensor(pos) for pos in positions], batch_first=True, padding_value=129)
    labels = pad_sequence([torch.tensor(lab.astype(np.int64)) for lab in labels], batch_first=True, padding_value=129)
    
    return images, ground_truths, column_maps, positions, labels, dmaps, ldesc_maps

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

with open('test_exp_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

point_sets = test_data['points']
images = test_data['images']
preprocessed_images = []
exp_density_maps = []
exp_ldesc_maps = []
for idx, image in enumerate(images):
    normalized_image = np.maximum((image - image.min()) / (image.max() - image.min()), 0)
    normalized_image = normalized_image[np.newaxis,np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
    preprocessed_images.append(image_tensor)
    pos = point_sets[idx]
    tri = Delaunay(pos)
    densities = np.zeros(len(pos))
    for i in range(len(pos)):
        densities[i] = calculate_density(tri, i, pos)
    grid_x, grid_y = np.mgrid[0:128, 0:128]
    density_map = griddata(pos, densities, (grid_x, grid_y), method='cubic', fill_value=0)
    density_map = density_map.T
    density_map = density_map[np.newaxis,np.newaxis, :, :]
    density_map = torch.tensor(density_map, dtype=torch.float32)
    print(density_map.shape)
    exp_density_maps.append(density_map)
    ldesc_map = calc_lattice_descriptor_map(idx, point_sets)
    ldesc_map = ldesc_map[np.newaxis,np.newaxis, :, :]
    ldesc_map = torch.tensor(ldesc_map, dtype=torch.float32)
    exp_ldesc_maps.append(ldesc_map)
    

exp_images = preprocessed_images

column_maps = test_data['column_maps']
predictions = [torch.tensor(np.where(column_map >= 0.01, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32) for column_map in column_maps]
exp_column_maps = predictions
exp_labels = test_data['labels']
exp_point_sets=point_sets


image,column_map, ground_truth,positions,labels, density_map, ldesc_map = validation_dataset[0]
# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = seg_UNet().to(device)
#criterion= DiceLoss()
criterion = nn.BCELoss()
#criterion = iou_loss
optimizer = optim.Adam(model.parameters(), lr=0.000005)  # Starting learning rate is set to 0.1

# Define the scheduler

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies= []
particle_accuracies = []
test_accuracies = []
tot_accuracies = []
num_epochs = 150
best_acc = 0.0
best_loss = np.inf
best_val_acc = 0.0
val_at_best_test = np.inf
checkp = 0

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.000001)
from torch.optim.lr_scheduler import CyclicLR
scheduler = CyclicLR(optimizer, base_lr=0.000005, max_lr=0.00001, step_size_up=5, step_size_down=5, cycle_momentum=False)

#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.00003)
for epoch in range(num_epochs):
    model.train()
    total_correct_train = 0
    total_predictions_train = 0
    total_correct_val = 0
    total_predictions_val = 0
    total_correct_1 = 0
    total_predictions_1 = 0
    running_train_accuracy = 0
    train_loss = 0.0

    for images, ground_truths, column_maps, positions, labels,density_maps,ldesc_maps in tqdm(train_dataloader):
        images = images.to(device)
        ground_truths = ground_truths.to(device)
        #print(column_maps)
        column_maps = column_maps.to(device)
        positions = positions.to(device)
        labels = labels.to(device)
        density_maps = density_maps.to(device)
        ldesc_maps = ldesc_maps.to(device)

        optimizer.zero_grad()
        #print(images.shape,column_maps.shape)
        #outputs = model(images, column_maps,density_maps,ldesc_maps)
        outputs = model(column_maps)
        predicted_regions = compute_regions(outputs)
        loss_bce = criterion(outputs, ground_truths)
        accuracy = 0
        tot_positions = 0
        for i in range(len(images)):
            positions_np = positions[i].cpu().numpy()
            positions_np = np.flip(positions_np,1)
            labels_np = labels[i].cpu().numpy()
            #print(len(positions_np),len(labels_np))
            # Create mask to ignore padded elements
            mask_pos = ~np.all(np.isclose(positions_np, 129.0), axis=-1)
            mask_lab = ~np.isclose(labels_np, 129.0)
            
            masked_positions = positions_np[mask_pos] - 0.5
            masked_labels = labels_np[mask_lab]
            #print(masked_positions.shape,masked_labels.shape)
            #if i==0 and epoch == 6:
            #    plt.imshow(predicted_regions[i].cpu().numpy().squeeze())
            #    plt.scatter(masked_positions[:,1],masked_positions[:,0])
            #    plt.show()
            pred_labels = label_points(masked_positions, predicted_regions[i].cpu().numpy())
            total_correct_train += accuracy_score(masked_labels, pred_labels, normalize=False)
            total_predictions_train += len(masked_positions)
            
        #print('batch accuracy',accuracy/tot_positions)
        #accuracy_loss = 1 - accuracy / len(images)
        loss = loss_bce
        loss.backward()
        optimizer.step()

        train_loss += loss_bce.item()
        running_train_accuracy += accuracy / len(images)

    train_loss /= len(train_dataloader)
    train_accuracy = running_train_accuracy / len(train_dataloader)
    train_acc = total_correct_train / total_predictions_train
    #print('train accuracy',train_acc)
    train_losses.append(train_loss)
    #print(train_accuracy)
    #train_accuracies.append(train_accuracy)

    model.eval()
    running_val_accuracy = 0
    validation_loss = 0.0
    image, ground_truth, column_map, output = None, None, None, None

    for i, (images, ground_truths, column_maps, positions, labels, density_maps, ldesc_maps) in enumerate(tqdm(validation_dataloader)):
        images = images.to(device)
        ground_truths = ground_truths.to(device)
        column_maps = column_maps.to(device)
        positions = positions.to(device)
        labels = labels.to(device)
        density_maps = density_maps.to(device)
        ldesc_maps = ldesc_maps.to(device)

        with torch.no_grad():
            #print(images.shape,column_maps.shape)
            #outputs = model(images, column_maps,density_maps,ldesc_maps)
            outputs = model(column_maps)
            #outputs = torch.where(outputs > 0.5, torch.tensor(1.0, device=outputs.device), torch.tensor(0.0, device=outputs.device))
            loss_bce = criterion(outputs, ground_truths)
            predicted_regions = compute_regions(outputs)
            accuracy = 0
            for j in range(len(images)):
                positions_np = positions[j].cpu().numpy()
                positions_np = np.flip(positions_np,1)
                labels_np = labels[j].cpu().numpy()
                
                # Create mask to ignore padded elements
                mask_pos = ~np.all(np.isclose(positions_np, 129.0), axis=-1)
                mask_lab = ~np.isclose(labels_np, 129.0)
                # Apply mask to positions, labels, and pixel_coordinates
                masked_positions = positions_np[mask_pos] - 0.5
                #print(masked_positions)
                #print(masked_positions.shape,positions[i].shape)
                masked_labels = labels_np[mask_lab]
                #total_predictions += len(masked_labels)
                total_predictions_1 += np.sum(masked_labels)

                pred_labels = label_points(masked_positions, predicted_regions[j].cpu().numpy())
                total_correct_val += accuracy_score(masked_labels, pred_labels, normalize=False)
                total_predictions_val += len(masked_positions)
                correct = (pred_labels == masked_labels)
                num_correct = np.sum(correct)
                #total_correct += num_correct

                correct_1 = (pred_labels == masked_labels) & (masked_labels == 1)
                num_correct_1 = np.sum(correct_1)
                total_correct_1 += num_correct_1

            accuracy_loss = 1 - accuracy / len(images)
            loss = loss_bce
            running_val_accuracy += accuracy / len(images)
            validation_loss += loss_bce.item()

            if i == 0:
                image = images[6].cpu().detach().numpy()[0]
                ground_truth = ground_truths[6].cpu().detach().numpy()[0]
                column_map = column_maps[6].cpu().detach().numpy()[0]
                ldesc_map = ldesc_maps[6].cpu().detach().numpy()[0]
                output = outputs[6].cpu().detach().numpy()[0]

    validation_loss /= len(validation_dataloader)
    val_acc = total_correct_val / total_predictions_val
    #for each image in test_data, compute only the total accuracy. test_data has 10 samples with image,column_map,point_set and corresponding label
    correct_class = 0
    total_points = 0
    for im,col_map,positions,labels,dens_map, ld_map in zip(exp_images, exp_column_maps, exp_point_sets, exp_labels, exp_density_maps, exp_ldesc_maps):
        with torch.no_grad():
            #print('hello')
            #print(image.shape,column_map.shape)
            im = im.to(device)
            col_map = col_map.to(device)
            dmap = dens_map.to(device)
            ld_map = ld_map.to(device)
            #out = model(im, col_map,dmap,ld_map)
            out = model(col_map)
            predicted_region = compute_regions(out)
            predicted_region = predicted_region.squeeze(0).squeeze(0).cpu().numpy()
            #print(positions.shape, predicted_region.shape)
            pixel_coordinates = np.floor(positions).astype(int)
            point_labels = []
            for pixel_coord in pixel_coordinates:
                point_labels.append(predicted_region[pixel_coord[0], pixel_coord[1]])
            pred_labels = np.array(point_labels)

            #print(pred_labels)
            correct = (pred_labels == labels)
            #print(correct)
            num_correct = np.sum(correct)
            #print(num_correct)
            correct_class += num_correct
            total_points += len(labels)
            #print(correct_class,total_points)
    test_accuracy = correct_class/total_points
    test_accuracies.append(test_accuracy)

    #overall_accuracy = total_correct / total_predictions
    overall_accuracy_1 = total_correct_1 / total_predictions_1
    
    #tot_accuracies.append(overall_accuracy)
    particle_accuracies.append(overall_accuracy_1)

    val_accuracy = running_val_accuracy / len(validation_dataloader)

    validation_losses.append(validation_loss)

    train_accuracies.append(train_acc)
    validation_accuracies.append(val_acc)

    if test_accuracy >= best_acc and validation_loss < val_at_best_test:
        best_acc = test_accuracy
        val_at_best_test = validation_loss
        checkp = epoch
        # Save the model state dictionary, best accuracy, and epoch number
        checkpoint = {
            'epoch': epoch,
            'best_accuracy': best_acc,
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, 'best_test_acc_gaussian_2.pth')

    if validation_loss < best_loss:
        best_loss = validation_loss
        checkpoint = {
            'epoch': epoch,
            'best_loss': best_loss,
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, 'best_val_loss_gaussian_2.pth')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint = {
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, 'best_val_acc_gaussian_2.pth')
        
    curr_lr = optimizer.param_groups[0]['lr']
    print('total_correct_train',total_correct_train,'total_predictions_train',total_predictions_train,'best_acc',best_acc, checkp)
    print(f'Epoch: {epoch + 1} \tLearning Rate: {curr_lr:.9f} \tTest accuracy: {test_accuracy:.4f} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {validation_loss:.4f} \tVal Accuracy: {val_acc:.4f} \tTrain Accuracy: {train_acc:.4f}')
    print("Current lr: ", curr_lr)
    #scheduler.step()
    #plot only every 10 epoch
    if epoch % 10 == 0:
        plt.subplot(2,2,1)
        plt.imshow(image, cmap='gray')
        plt.title('Image')
        plt.axis('off')

        plt.subplot(2,2,2)
        plt.imshow(column_map, cmap='gray')
        plt.title('Column Map')
        plt.axis('off')

        plt.subplot(2,2,3)
        plt.imshow(ground_truth, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(2,2,4)
        plt.imshow(output, cmap='gray')
        plt.title('Output')
        plt.axis('off')

        plt.show() 
#save losses and accuracies
with open('train_losses_gaussian_2.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(train_losses, f)
with open('train_accuracies_gaussian_2.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(train_accuracies, f)
with open('validation_losses_gaussian_2.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(validation_losses, f)
with open('validation_accuracies_gaussian_2.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(validation_accuracies, f)
with open('test_accuracies_gaussian_2.pkl', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(test_accuracies, f)


# Plotting the losses
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting the accuracies
plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs+1), validation_accuracies, label='Validation Accuracy')
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting the accuracies
plt.subplot(1, 3, 3)
plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print("Best test accuracy: ", best_acc, "Best validation accuracy: ", best_val_acc, "Best loss: ", best_loss)

# %%
#seg_data = torch.load("magiskt_bra.pth")

checkpoint = torch.load('best_test_acc_2_save_grym.pth')
seg_data = checkpoint['model_state_dict']
segmenter = seg_UNet()

from PI_U_Net import UNet
from Analyzer import Analyzer

localizer = UNet()
loc_data = torch.load("best_model_data.pth")
loaded_model_state_dict = loc_data['model_state_dict']
localizer.load_state_dict(loaded_model_state_dict)

#loaded_validation_loss = loaded_data['validation_loss']
segmenter.load_state_dict(seg_data)
#segmenter.load_state_dict(checkpoint['model_state_dict'])

image_path = "data/experimental_data/32bit/10.tif"
image_tensor = preprocess_image(image_path)
image_tensor = image_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
localizer = localizer.cuda()
# Pass the image tensor through the model
localizer.eval()
with torch.no_grad():
    predicted_output = localizer(image_tensor)

predicted_localization = postprocess_output(predicted_output)
prediction = torch.tensor(np.where(predicted_localization >= 0.12, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
predicted_localization_save = predicted_localization.copy()
analyzer = Analyzer()
pred_positions = analyzer.return_positions_experimental(predicted_localization)+[0.5,0.5]
#set all values larger than 0.1 in predicted_localization to 1
predicted_localization[predicted_localization > 0.12] = 1
input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()

segmenter = segmenter.cuda()
segmenter.eval()

pos = pred_positions
tri = Delaunay(pos)
densities = np.zeros(len(pos))
for i in range(len(pos)):
    densities[i] = calculate_density(tri, i, pos)
grid_x, grid_y = np.mgrid[0:128, 0:128]
density_map = griddata(pos, densities, (grid_x, grid_y), method='cubic', fill_value=0)
density_map = density_map
density_map = density_map[np.newaxis,np.newaxis, :, :]
density_map = torch.tensor(density_map, dtype=torch.float32)
ld_map = calc_lattice_descriptor_map(0, [pos]).T
ld_map = ld_map[np.newaxis,np.newaxis, :, :]
ld_map = torch.tensor(ld_map, dtype=torch.float32)
#predicted_output[predicted_output > 0.01] = 1  
with torch.no_grad():
    print(image_tensor.shape,predicted_output.shape,density_map.shape)
    #predicted_output = segmenter(image_tensor, predicted_output,density_map.to(device),ld_map.to(device))
    predicted_output = segmenter(prediction.to(device))
predicted_segmentation = postprocess_output(predicted_output)

# Convert predicted segmentation to binary using threshold of 0.5
binary_segmentation = np.where(predicted_segmentation > 0.5, 1, 0)

labels = label_points(pred_positions, binary_segmentation)
plt.figure(figsize=(15, 5))
plt.imshow(input_image_np, cmap='gray')
plt.axis('off')
plt.show()
plt.figure(figsize=(15, 5))
plt.imshow(predicted_localization_save, cmap='gray')
plt.axis('off')
plt.show()
plt.figure(figsize=(15, 5))
plt.imshow(binary_segmentation, cmap='gray')
plt.axis('off')
plt.show()
plt.figure(figsize=(15, 5))
plt.imshow(input_image_np, cmap='gray')
for i in range(len(pos)):
    if labels[i] == 1:
        plt.scatter(pos[i][1],pos[i][0], s=40, c='springgreen')
    else:
        plt.scatter(pos[i][1],pos[i][0], s=40, c='darkorange')
plt.axis('off')
plt.show()
plt.figure(figsize=(5,5))
plt.scatter(pos[:,1],pos[:,0], s=40,c='k')
plt.xlim(0,128)
plt.ylim(0,128)
plt.gca().invert_yaxis()
plt.gca().set_xticks([])
plt.gca().set_yticks([])
#plt.axis('equal')
#plt.axis('off')
plt.show()
# Plot the input image and predicted segmentation
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(input_image_np, cmap='gray')
plt.axis('off')
plt.title('Input Image')

plt.subplot(1, 4, 2)
plt.imshow(predicted_localization, cmap='gray')
plt.axis('off')
plt.title('Predicted Localization')

plt.subplot(1, 4, 3)
plt.imshow(predicted_segmentation, cmap='gray')
plt.axis('off')
plt.title('Predicted Segmentation')

plt.subplot(1, 4, 4)
plt.imshow(binary_segmentation, cmap='gray')
plt.axis('off')
plt.title('Binary Segmentation')

plt.show()

#plot input image, predicted localization save, density map, binary segmentation, and Input image with pos on top colored by label
plt.figure(figsize=(15, 5))

plt.subplot(1, 6, 1)
plt.imshow(input_image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 2)
plt.imshow(prediction.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 3)
plt.imshow(input_image_np, cmap='gray') # add the origin parameter
plt.triplot(pos[:,1], pos[:,0], tri.simplices)
plt.axis('off')

plt.subplot(1, 6, 4)
plt.imshow(ld_map.squeeze(0).squeeze(0).cpu().numpy(), cmap='hot', interpolation='nearest') # add the origin parameter
plt.axis('off')

plt.subplot(1, 6, 5)
plt.imshow(binary_segmentation, cmap='gray')
plt.axis('off')
#color by label, 1 is blue,0 is red


plt.subplot(1, 6, 6)
plt.imshow(input_image_np, cmap='gray')
for i in range(len(pos)):
    if labels[i] == 1:
        plt.scatter(pos[i][1],pos[i][0], s=10, c='c')
    else:
        plt.scatter(pos[i][1],pos[i][0], s=10, c='orange')
plt.axis('off')

plt.show()

#plot triangulation
#plt.figure(figsize=(10, 5))
#plt.imshow(input_image_np, cmap='gray') # add the origin parameter
#plt.triplot(pos[:,1], pos[:,0], tri.simplices)
#plt.axis('off')
#plt.show()

torch.cuda.empty_cache()

# %%
#Follow a single simulated example
from Data_Generator import Data_Generator
from PI_U_Net import UNet
from Analyzer import Analyzer
import pandas as pd
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN

#seg_data = torch.load("magiskt_bra.pth")
checkpoint = torch.load('best_test_acc_2_save_grym.pth')
seg_data = checkpoint['model_state_dict']
im_idx = 5
segmenter = seg_UNet()
localizer = UNet()
loc_data = torch.load("best_model_data.pth")
loaded_model_state_dict = loc_data['model_state_dict']
localizer.load_state_dict(loaded_model_state_dict)
segmenter.load_state_dict(seg_data)

#number_of_images = 2
#dataset_name = "paper_img_xd"
#generator = Data_Generator()
#generator.generate_data(number_of_images,dataset_name,benchmark=True)
with open('paper_img_xd.pkl', 'rb') as f:
    data = pickle.load(f)
with open('benchmark_100_3.pkl', 'rb') as f:
    data = pickle.load(f)
raw_image = data['images'][im_idx]
normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
normalized_image = normalized_image[np.newaxis, :, :]
image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
image_tensor = image_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
localizer = localizer.cuda()
# Pass the image tensor through the model
localizer.eval()
with torch.no_grad():
    predicted_output = localizer(image_tensor)
predicted_localization = postprocess_output(predicted_output)
predicted_localization_save = predicted_localization.copy()
prediction = torch.tensor(np.where(predicted_localization >= 0.12, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
analyzer = Analyzer()
pred_positions = analyzer.return_positions_experimental(predicted_localization)+[0.5,0.5]

analyzer.set_image_list(data["images"])
analyzer.set_gt_list(data['exponential_ground_truths'])
analyzer.set_pred_list([predicted_localization_save])
#print('boi',data["dataframes"])
analyzer.set_point_sets(data["dataframes"])
#mean, std = analyzer.calculate_average_error()

intersections,centers,radii,bbox_image,minr,minc,maxr,maxc,gt_pos=analyzer.plot_how_it_works(predicted_localization_save,25,im_idx)
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(bbox_image, cmap='gray',origin='lower')

# Add circles
for center, radius in zip(centers, radii):
    circle = plt.Circle((center[1] - minc + 0.5, center[0] - minr + 0.5), radius, fill=False, edgecolor='darkorange', linewidth=4)
    ax.add_artist(circle)

if len(intersections) > 0:
    plt.scatter(intersections[:, 1]+0.5, intersections[:, 0]+0.5, marker='x',s=50, color='springgreen')
    plt.scatter(gt_pos[0]-minc+0.5, gt_pos[1]+0.5-minr, marker='o', s=60,color='red')
plt.xlim([0, maxc - minc])
plt.ylim([0, maxr - minr])
plt.axis('equal')
plt.axis('off')
plt.show()
gt = predicted_localization_save
fig, ax = plt.subplots()
ax.imshow(gt, cmap='gray',origin='lower')
# Create a Rectangle patch
print(maxc,minc,maxr,minr)
rect = patches.Rectangle((minc-2.5, minr-2.5), maxc - minc+4, maxr - minr+4, linewidth=4, edgecolor='darkorange', facecolor='none')
# Add the rectangle to the plot
ax.add_patch(rect)
ax.axis('off')
#ax.axis('equal')
# Show the figure
plt.show()

dbscan = DBSCAN(eps=0.1, min_samples=3)
dbscan.fit(intersections) 
labels = dbscan.labels_
clusters = [intersections[labels == i] for i in range(max(labels) + 1)]
clusters = sorted(clusters, key=len, reverse=True)

#get the brightest pixel in the bbox_image
flat_index = np.argmax(bbox_image)
row, col = np.unravel_index(flat_index, bbox_image.shape)

cluster_means = [np.mean(cluster, axis=0) for cluster in clusters]

def point_within_pixel(point, pixel):
    px_x, px_y = pixel
    pt_x, pt_y = point

    if pt_x >= px_x and pt_x <= px_x + 1 and pt_y >= px_y and pt_y <= px_y + 1:
        return True
    else:
        return False 
#------------------------------------------------------------------------------
# #implement in Analyzer + maybe look within 1 pixel euclidian distance instead of only in brightest pixel.    
# Find the index of the largest cluster whose mean lies within [row, col]
index = None
mean_value = None
print(cluster_means)
for i, mean in enumerate(cluster_means):
    print(mean,row,col)
    if point_within_pixel(mean+0.5, np.array([row, col])-0.5):
        print('yes')
        index = i
        mean_value = mean+0.5
        break
if mean_value is None:
    mean_value = np.array([row, col])
#------------------------------------------------------------------------------
print(mean_value)
predicted_column_position = mean_value
plt.figure()
plt.imshow(bbox_image,cmap='gray',origin='lower')
plt.scatter(intersections[:, 1]+0.5, intersections[:, 0]+0.5, marker='x',s=50, color='springgreen')
plt.scatter(np.array(cluster_means)[:, 1]+0.5, np.array(cluster_means)[:, 0]+0.5,s=50, marker='o', color='darkorange')
plt.scatter(predicted_column_position[1],predicted_column_position[0], s=60, c='k',marker='x')
plt.scatter(gt_pos[0]-minc+0.5, gt_pos[1]+0.5-minr, s=60, c='red',marker='o')
#plt.scatter(gt_pos[0]-minc+0.5, gt_pos[1]+0.5-minr, s=20, c='red')
plt.axis('off')
plt.show()

#set all values larger than 0.1 in predicted_localization to 1
predicted_localization[predicted_localization > 0.12] = 1
input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()

segmenter = segmenter.cuda()
segmenter.eval()
with torch.no_grad():
    print(image_tensor.shape,predicted_output.shape,density_map.shape)
    #predicted_output = segmenter(image_tensor, predicted_output,density_map.to(device),ld_map.to(device))
    predicted_output = segmenter(prediction.to(device))

predicted_segmentation = postprocess_output(predicted_output)

# Convert predicted segmentation to binary using threshold of 0.5
binary_segmentation = np.where(predicted_segmentation > 0.5, 1, 0)

boi = binary_segmentation+predicted_localization*data["segmented_images"][im_idx]
boi = np.where(boi ==2, 0,boi)
binary_temp = np.array(binary_segmentation).astype(np.uint8)
boi_diff= binary_temp^np.array(np.where(predicted_localization < 0.12,0,1)*data["segmented_images"][im_idx]).astype(np.uint8)

labels = label_points(pred_positions, binary_segmentation)
pos = pred_positions
plt.figure()
plt.imshow(input_image_np, cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_localization_save, cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_localization, cmap='gray', origin="lower")
plt.axis('off')
plt.show()
print("bob")
plt.figure()
plt.imshow(data["segmented_images"][im_idx], cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(data["segmented_images"][im_idx] * predicted_localization, cmap='gray', origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_segmentation, cmap='gray',origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(binary_segmentation, cmap='gray',origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(input_image_np, cmap='gray',origin="lower")
for i in range(len(pos)):
    if labels[i] == 1:
        plt.scatter(pos[i][1],pos[i][0], s=20, c='springgreen')
    else:
        plt.scatter(pos[i][1],pos[i][0], s=20, c='darkorange')
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_segmentation, cmap='gray',origin="lower")
for i in range(len(pos)):
    if labels[i] == 1:
        plt.scatter(pos[i][1]-0.5,pos[i][0]-0.5, s=20, c='springgreen')
    else:
        plt.scatter(pos[i][1]-0.5,pos[i][0]-0.5, s=20, c='darkorange')
#plt.axis('off')
plt.show()
plt.figure()
plt.imshow(predicted_segmentation, cmap='gray',origin="lower")
for i in range(len(pos)):
    plt.scatter(pos[i][1]-0.5,pos[i][0]-0.5, s=20, c='darkorange')
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(boi_diff, cmap='gray',origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.scatter(pos[:,1],pos[:,0], s=40,c='k')
plt.xlim(0,128)
plt.ylim(0,128)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().set_aspect('equal', adjustable='box')
#plt.axis('off')
plt.show()
#plot the exponential ground truth and the gaussian ground truth in separete figures
plt.figure()
plt.imshow(data['exponential_ground_truths'][im_idx], cmap='gray',origin="lower")
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(data['gaussian_ground_truths'][im_idx], cmap='gray',origin="lower")
plt.axis('off')
plt.show()
#%%
import hyperspy.api as hs
import atomap.api as am
import statistics

image_data = data["benchmark_sets"]
#print(image_data)

#extract x and y coordinates from dataframe
positions_gt = data["dataframes"][im_idx][["x","y"]].to_numpy()*128
positions_gt = positions_gt[:,::-1]
positions_gt = positions_gt[
    np.logical_and.reduce((
        positions_gt[:,0] > 5, 
        positions_gt[:,0] < 123, 
        positions_gt[:,1] > 5, 
        positions_gt[:,1] < 123
    ))]

def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


unet_positions = []
unet_positions_gaussian_fitting = []
unet_positions_gaussian_com = []

unet_diff = []
unet_diff_gaussian = []

boi = []

atomaps_diff = []
import math
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def closest_pairs(set_a, set_b):
    set_a = [tuple(point) for point in set_a]
    set_b = [tuple(point) for point in set_b]
    
    closest_to_a = {}
    for point_a in set_a:
        min_distance = float('inf')
        closest_point = None
        for point_b in set_b:
            dist = euclidean_distance(point_a, point_b)
            if dist < min_distance and dist < 3:
                min_distance = dist
                closest_point = point_b
        if closest_point is not None:
            if closest_point not in closest_to_a or min_distance < euclidean_distance(closest_point, closest_to_a[closest_point]):
                closest_to_a[closest_point] = point_a

    # Points in set_a that don't have a close point in set_b
    unpaired_a = [point for point in set_a if point not in closest_to_a.values()]
    # Points in set_b that haven't been assigned
    unassigned_b = [point for point in set_b if point not in closest_to_a.keys()]
    #print(len(unpaired_a),len(unassigned_b))
    return unpaired_a + unassigned_b, unpaired_a, unassigned_b



from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass
for i, group in enumerate(image_data):
    if i == im_idx:
        for j, image in enumerate(group):
            raw_image = image
            normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
            normalized_image = normalized_image[np.newaxis, :, :]
            image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image_tensor = image_tensor.to(device)
            localizer = localizer.cuda()
            # Pass the image tensor through the model
            localizer.eval()
            with torch.no_grad():
                predicted_output = localizer(image_tensor)
            predicted_localization = postprocess_output(predicted_output)
            predicted_localization_save = predicted_localization.copy()
            prediction = torch.tensor(np.where(predicted_localization >= 0.01, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
            analyzer = Analyzer()
            pred_positions = analyzer.return_positions_experimental(predicted_localization)+[0.5,0.5]

            unet_positions.append(pred_positions)
            #ignore points close to edge
            pred_positions = pred_positions[
                np.logical_and.reduce((
                    pred_positions[:,0] > 5, 
                    pred_positions[:,0] < 123, 
                    pred_positions[:,1] > 5, 
                    pred_positions[:,1] < 123
                ))
            ]
            diff, fp, fn = closest_pairs(pred_positions,positions_gt)
            unet_diff.append(len(diff))

            #Gaussian fitting
            localizer_gaussian = UNet()
            loc_data = torch.load("model_data_epoch_gaussian72.pth")
            loaded_model_state_dict = loc_data['model_state_dict']
            localizer_gaussian.load_state_dict(loaded_model_state_dict)
            localizer_gaussian = localizer_gaussian.cuda()
            localizer_gaussian.eval()
            with torch.no_grad():
                predicted_output = localizer_gaussian(image_tensor)
            predicted_localization = postprocess_output(predicted_output)
            predicted_localization_save = predicted_localization.copy()
            prediction_gaussian = torch.tensor(np.where(predicted_localization >= 0.1, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
            prediction_gaussian = prediction_gaussian.squeeze(0).squeeze(0).cpu().numpy()

            #CoM
            labeled_image = label(prediction_gaussian)
            regions = regionprops(labeled_image)
            centroids = [region.centroid for region in regions]
            centroids = np.array(centroids)
            centers_of_mass_gaussian = []
            for region in regions:
                # Create a binary mask for this region
                mask = (labeled_image == region.label)
                # Extract the region's pixel values from the grayscale image
                region_values = mask * predicted_localization_save
                # Compute the center of mass for the region using the grayscale values
                center = center_of_mass(region_values)
                centers_of_mass_gaussian.append(center)
            unet_positions_gaussian_fitting.append(centers_of_mass_gaussian)
            centers_of_mass_gaussian = np.array(centers_of_mass_gaussian)
            centers_of_mass_gaussian = centers_of_mass_gaussian[
                np.logical_and.reduce((
                    centers_of_mass_gaussian[:,0] > 5, 
                    centers_of_mass_gaussian[:,0] < 123, 
                    centers_of_mass_gaussian[:,1] > 5, 
                    centers_of_mass_gaussian[:,1] < 123
                ))
            ]
            print(centers_of_mass_gaussian.shape, positions_gt.shape, centers_of_mass_gaussian.dtype, positions_gt.dtype)
            diff_g, fp_g, fn_g = closest_pairs(centers_of_mass_gaussian,positions_gt)
            print(diff_g,centers_of_mass_gaussian,positions_gt)
            unet_diff_gaussian.append(len(diff_g))
            boi.append(diff)
            if j == 12:
                plt.figure()
                plt.scatter(centers_of_mass_gaussian[:,1],centers_of_mass_gaussian[:,0], s=10, c='springgreen')
                plt.scatter(positions_gt[:,1],positions_gt[:,0], s=10, c='darkorange')
                plt.show()
                plt.figure()
                plt.imshow(labeled_image, cmap='gray', origin="lower")
                plt.show()   
                diff_g = np.array([list(x) for x in diff_g])
                #temp = np.array([list(t) for t in unet_positions_gaussian_com[10]])
                plt.figure()
                plt.imshow(predicted_localization_save, cmap='gray', origin="lower")
                plt.scatter(diff_g[:,1],diff_g[:,0], s=20, c='springgreen')
                plt.show()

            subpixel_centroids = []
            patch_size = 3 # only odd numbers please
            from scipy.optimize import curve_fit
            for region in regions:
                y0, x0 = region.centroid
                y0, x0 = int(y0), int(x0)
                # Ensure the patch stays within the image bounds
                y_start = max(0, y0 - patch_size)
                y_end = min(prediction_gaussian.shape[0], y0 + patch_size+2)
                x_start = max(0, x0 - patch_size)
                x_end = min(prediction_gaussian.shape[1], x0 + patch_size+2)
                
                patch = predicted_localization_save[y_start:y_end, x_start:x_end]
                #plt.imshow(patch, cmap='gray', origin="lower")
                #plt.show()
                x = np.linspace(0, patch.shape[1]-1, patch.shape[1])
                y = np.linspace(0, patch.shape[0]-1, patch.shape[0])
                x, y = np.meshgrid(x, y)

                # Initial guess for Gaussian parameters:
                initial_params = (patch.max(), patch_size, patch_size, 1, 1, 0, patch.min())
                
                try:
                    popt, _ = curve_fit(twoD_Gaussian, (x, y), patch.ravel(), p0=initial_params)
                    subpixel_centroids.append((y0-patch_size+popt[2], x0-patch_size+popt[1]))
                except:
                    # If fitting fails for some reason, use original centroid
                    subpixel_centroids.append((y0, x0))
            unet_positions_gaussian_com.append(subpixel_centroids)


atomaps_positions = []
for i, group in enumerate(image_data):
    if i == im_idx:
        for j, image in enumerate(group):
            image_data = image.T
            s = hs.signals.Signal2D(image_data)

            if j == 0:
                #paper, S1
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 3, threshold_rel=0.2)
                sublattice = am.Sublattice(atom_position_list=atom_positions, image=s.data)
                sublattice.construct_zone_axes()
                sublattice.refine_atom_positions_using_center_of_mass(sublattice.image)  
                sublattice.refine_atom_positions_using_2d_gaussian(sublattice.image)  
                sublattice.plot()   
            if j == 1:
                #paper, S1
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 3, threshold_rel=0.2)
            if j == 2:
                #paper
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.3)
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 3, threshold_rel=0.3)
                #s2
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)
                #s3
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 3, threshold_rel=0.2)
            if j == 3:
                #paper
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.15)
                #atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.4)
                #atom_positions = am.get_atom_positions(s, 3, threshold_rel=0.4)               
                #s2
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)
                #s3
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)           
            if j == 4:
                #paper
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)
                #s1
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)
            if j == 5:
                #paper
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)          
                #s3
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.3)      
            if j == 6:
                #paper
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.3)
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)     
                #s2
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.3) 
                #s3
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.3)               
            if j == 7:
                #paper
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s,6, threshold_rel=0.3)
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.2)   
                #s2
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.3)          
                #s3
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.15)
                atom_positions = am.get_atom_positions(s, 6, threshold_rel=0.15)   
            if j == 8:
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 6, threshold_rel=0.3)   
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 4, threshold_rel=0.3)     
                #s2
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)                            
                #s3
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.3)        
            if j == 9:
                #s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 6, threshold_rel=0.3)
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.3)
                #s2
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)                        
                #s3
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)           
            if j == 10:
                #s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 6, threshold_rel=0.3)
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.3)                    
                #s2
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)                             
                #s3
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.2)
                atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.2)
                sublattice = am.Sublattice(atom_position_list=atom_positions, image=s.data)
                sublattice.construct_zone_axes()
                sublattice.refine_atom_positions_using_center_of_mass(sublattice.image)  
                sublattice.refine_atom_positions_using_2d_gaussian(sublattice.image)  
                sublattice.plot()        
            if j == 11:
                #s_separation = am.get_feature_separation(s, separation_range=(4,7), threshold_rel=0.28)
                #atom_positions = am.get_atom_positions(s, 8, threshold_rel=0.28)
                #S1
                #s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                #atom_positions = am.get_atom_positions(s, 5, threshold_rel=0.3)               
                #s2
                s_separation = am.get_feature_separation(s, separation_range=(3,7), threshold_rel=0.3)
                atom_positions = am.get_atom_positions(s, 6, threshold_rel=0.3)                    
            print(j)
            sublattice = am.Sublattice(atom_position_list=atom_positions, image=s.data)
            sublattice.construct_zone_axes()
            sublattice.refine_atom_positions_using_center_of_mass(sublattice.image)  
            sublattice.refine_atom_positions_using_2d_gaussian(sublattice.image)  
            #sublattice.plot()
            positions_atomaps = sublattice.atom_positions
            if j == 0:
                tot = len(positions_atomaps)
            atomaps_positions.append(positions_atomaps)
            #ignore points close to edge
            positions_atomaps = positions_atomaps[
                np.logical_and.reduce((
                    positions_atomaps[:,0] > 5, 
                    positions_atomaps[:,0] < 123, 
                    positions_atomaps[:,1] > 5, 
                    positions_atomaps[:,1] < 123
                ))
            ]
            diff, fp, fn = closest_pairs(positions_atomaps,positions_gt)            
            atomaps_diff.append(len(diff))
            print(len(diff))
            print(atomaps_diff)

#%%
unet_errors = []
atomaps_errors = []
unet_gaussian_errors = []
unet_gaussian_com_errors = []
#print(unet_positions)
plt.figure()
plt.scatter(unet_positions[0][:,0],unet_positions[0][:,1], s=40,c='r')
plt.scatter(atomaps_positions[0][:,0],atomaps_positions[0][:,1], s=40,c='b')
#plt.scatter(positions_gt[:,0],positions_gt[:,1], s=40,c='k')
plt.show()
print(unet_positions_gaussian_com)

for u_p, u_p_g_f, u_p_g_c, a_p in zip(unet_positions,unet_positions_gaussian_fitting,unet_positions_gaussian_com,atomaps_positions):
    print("hello")
    nn_unet = analyzer.assign_nearest(u_p, positions_gt,3)
    nn_atomaps = analyzer.assign_nearest(a_p, positions_gt,3)
    nn_unet_gaussian = analyzer.assign_nearest(u_p_g_f, positions_gt,3)
    nn_unet_gaussian_com = analyzer.assign_nearest(u_p_g_c, positions_gt,3)
    
    unet_error = analyzer.calculate_error(u_p,nn_unet)
    atomaps_error = analyzer.calculate_error(a_p,nn_atomaps)
    unet_gaussian_error = analyzer.calculate_error(u_p_g_f,nn_unet_gaussian)
    unet_gaussian_com_error = analyzer.calculate_error(u_p_g_c,nn_unet_gaussian_com)
    unet_errors.append(unet_error)
    atomaps_errors.append(atomaps_error)
    unet_gaussian_errors.append(unet_gaussian_error)
    unet_gaussian_com_errors.append(unet_gaussian_com_error)
print("hello")
#%%
print(unet_errors,atomaps_errors)
print(boi)


#unet_errors,atomaps_errors = analyzer.compare_unet_atomaps(unet_positions,atomaps_positions, positions_gt)
mean_values_u = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
standard_deviations_u = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
mean_values_u_g_f = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_gaussian_errors])
standard_deviations_u_g_f = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_gaussian_errors])
mean_values_u_g_com = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_gaussian_com_errors])
standard_deviations_u_g_com = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_gaussian_com_errors])
mean_values_a = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
standard_deviations_a = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
ssims = data["benchmark_ssims"][im_idx]
#%%
#import MultipleLocator
from matplotlib.ticker import MultipleLocator
plt.figure()
plt.scatter(positions_gt[:,0],positions_gt[:,1], s=40,c='k')
plt.scatter(atomaps_positions[11][:,0],atomaps_positions[11][:,1], s=40,c='r')
plt.scatter(unet_positions[11][:,0],unet_positions[11][:,1], s=40,c='b')
plt.xlim(0,128)
plt.ylim(0,128)
fig, ax1 = plt.subplots()
print(mean_values_u,mean_values_a)
# Plotting data on primary y-axis
ax1.errorbar(ssims, mean_values_a, standard_deviations_a, linestyle='-', marker='s', color='darkorange',capsize=2,capthick=2, markerfacecolor='darkorange', markeredgecolor='black')
ax1.errorbar(ssims, mean_values_u, standard_deviations_u, linestyle='-', marker='o', color='darkgreen',capsize=2,capthick=2, markerfacecolor='darkgreen', markeredgecolor='black')
#ax1.errorbar(ssims, mean_values_u_g_f, standard_deviations_u_g_f, linestyle='-', marker='o', color='darkblue',capsize=2,capthick=2, markerfacecolor='darkblue', markeredgecolor='black')
ax1.errorbar(ssims, mean_values_u_g_com, standard_deviations_u_g_com, linestyle='-', marker='o', color='darkred',capsize=2,capthick=2, markerfacecolor='darkred', markeredgecolor='black')
ax1.set_ylabel('Mean error [pixels]', color='k',weight='bold')
ax1.tick_params(axis='y', labelcolor='k')
ax1.tick_params(axis='both', labelsize=11)
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

# Set major tick length larger and display labels, while setting minor tick length smaller and not displaying labels
ax1.tick_params(axis='y', which='major', length=10, labelsize=11)  # Adjust length and label size as needed
ax1.tick_params(axis='y', which='minor', length=5, labelsize=0)
ax1.set_xlabel('SSIM',weight='bold')
ax1.legend(['Atomaps','U-Net 3x3 gt','U-Net Gaussian gt, CoM'], facecolor='white', edgecolor='black', loc='upper right',title = 'Mean error',prop={'weight':'bold'})
# Create a twin y-axis
ax2 = ax1.twinx()

# Plotting data on secondary y-axis
ax2.plot(ssims, atomaps_diff, linestyle=':', color='gray')
ax2.plot(ssims, unet_diff, linestyle='-.', color='gray')
ax2.plot(ssims, unet_diff_gaussian, linestyle='--', color='k')

# Set secondary y-axis color
ax2.set_ylabel('# False predictions', color='gray',weight='bold')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.tick_params(axis='y', labelsize=11)
ax2.set_yticks(np.arange(0, 17, step=2))
ax2.legend(['Atomaps','U-Net 3x3 gt','U-Net Gaussian gt'], facecolor='white', edgecolor='black', loc=(0.6,0.49),title = 'False predictions',prop={'weight':'bold'})
#set figure title
plt.title('Benchmark',weight='bold')
plt.show()

#plot the 10th image and scatter the predicted unet_positions
temp = np.array([list(t) for t in unet_positions_gaussian_com[10]])
plt.figure()
plt.imshow(data["benchmark_sets"][im_idx][10], cmap='gray', origin="lower")
boii = np.array([list(t) for t in boi[10]])
#plt.scatter(boii[:,1],boii[:,0], s=40,c='g')
plt.scatter(unet_positions[10][:,1],unet_positions[10][:,0], s=40,c='b')
plt.scatter(temp[:,1],temp[:,0], s=40,c='r')
plt.scatter(positions_gt[:,1],positions_gt[:,0], s=40,c='g')
plt.axis('off')
plt.show()

#%%
unet_errors,atomaps_errors = analyzer.compare_unet_atomaps(unet_positions,atomaps_positions, positions_gt)
print(atomaps_errors[0])
#%%
import statistics
mean_values_u = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
standard_deviations_u = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
mean_values_a = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
standard_deviations_a = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
ssims = data["benchmark_ssims"][im_idx]

fig, ax1 = plt.subplots()

# Plotting data on primary y-axis
ax1.errorbar(ssims, mean_values_a, standard_deviations_a, linestyle='-', marker='s', color='darkorange',capsize=2,capthick=2, markerfacecolor='darkorange', markeredgecolor='black')
ax1.errorbar(ssims, mean_values_u, standard_deviations_u, linestyle='-', marker='o', color='darkgreen',capsize=2,capthick=2, markerfacecolor='darkgreen', markeredgecolor='black')
ax1.set_ylabel('Mean error [pixels]', color='k',weight='bold')
ax1.tick_params(axis='y', labelcolor='k')
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xlabel('SSIM',weight='bold')
ax1.legend(['Atomaps','U-Net'], facecolor='white', edgecolor='black', loc=(0.475,0.8),title = 'Mean error',prop={'weight':'bold'})
# Create a twin y-axis
ax2 = ax1.twinx()

# Plotting data on secondary y-axis
ax2.plot(ssims, atomaps_diff, linestyle=':', color='gray')
ax2.plot(ssims, unet_diff, linestyle='--', color='gray')

# Set secondary y-axis color
ax2.set_ylabel('# False predictions', color='gray',weight='bold')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.tick_params(axis='y', labelsize=11)
ax2.set_yticks(np.arange(0, 8, step=2))
ax2.legend(['Atomaps','U-Net'], facecolor='white', edgecolor='black', loc='upper right',title = 'False predictions',prop={'weight':'bold'})
plt.show()
#plot all images in data["benchmark_sets"][0] in a 3x4 grid with SSIM (two value digits) in the lower left of the image, i dont want any white space between images
plt.figure(figsize=(11,14.75))
for i, image in enumerate(data["benchmark_sets"][im_idx]):
    ax = plt.subplot(4, 3, i+1)
    ax.imshow(image, cmap='gray', origin="lower")
    ax.axis('off')
    ax.set_aspect('equal')  # Ensure the aspect ratio is equal
    # Use plt.text to place the SSIM on the image
    plt.text(5, 10, "SSIM: {:.2f}".format(data["benchmark_ssims"][im_idx][i]), 
             color='white', backgroundcolor='black', fontsize=15)

plt.subplots_adjust(wspace=0, hspace=0)  # Set spacing to zero
plt.show()

plt.figure(figsize=(14.75,11.15))
for i, image in enumerate(data["benchmark_sets"][im_idx]):
    ax = plt.subplot(3, 4, i+1)
    ax.imshow(image, cmap='gray', origin="lower")
    ax.axis('off')
    ax.set_aspect('equal')  # Ensure the aspect ratio is equal
    # Use plt.text to place the SSIM on the image
    plt.text(5, 10, "SSIM: {:.2f}".format(data["benchmark_ssims"][im_idx][i]), 
             color='white', backgroundcolor='black', fontsize=15)

plt.subplots_adjust(wspace=0, hspace=0)  # Set spacing to zero
plt.show()

plt.subplot(1,3,1)
plt.imshow(data["benchmark_sets"][im_idx][0], cmap='gray',origin="lower")
plt.scatter(unet_positions[0][:,1],unet_positions[0][:,0], s=10,c='r')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(data["benchmark_sets"][im_idx][1], cmap='gray',origin="lower")
plt.scatter(unet_positions[0][:,1],unet_positions[0][:,0], s=10,c='r')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(data["benchmark_sets"][im_idx][3], cmap='gray',origin="lower")
plt.scatter(atomaps_positions[3][:,1],atomaps_positions[3][:,0], s=10,c='r')
plt.axis('off')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(data["benchmark_sets"][im_idx][3], cmap='gray',origin="lower")
plt.axis('off')
plt.axis('equal')
plt.show()

#%%
import hyperspy.api as hs
import atomap.api as am
import statistics
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from scipy.optimize import curve_fit

means_u = []
stds_u = []

means_u_com = []
stds_u_com = []

means_u_gaussian_com = []
stds_u_gaussian_com = []

times_gaussian_fit = []
times_gaussian_com = []
times_exp = []

com_areas = []
exp_areas = []

mean_a = []
stds_a = []
ssims = []
tot_columns = []
tot_found_columns = []
differences = []

fp_exp = []
fp_gaussian = []
fn_exp = []
fn_gaussian = []

correct_predictions_3x3 = []
correct_predictions_gauss = []
tot_predictions = []

times_through_model_3x3 = []
times_through_model_gauss = []

confusion_matrices_3x3 = []
confusion_matrices_gauss = []

means_u_gaussian = []
stds_u_gaussian = []
differences_gaussian = []
localizer_gaussian_gt = UNet()
loc_data_gaussian = torch.load("model_data_epoch_gaussian72.pth")
loaded_model_state_dict_gauss = loc_data_gaussian['model_state_dict']
localizer_gaussian_gt.load_state_dict(loaded_model_state_dict_gauss)

checkpoint_exp = torch.load('best_test_acc_2_save_grym.pth')   
seg_data_exp = checkpoint_exp['model_state_dict']
checkpoint_gaussian = torch.load('best_test_acc_gaussian_2.pth')
seg_data_gaussian = checkpoint_gaussian['model_state_dict']
segmenter_exp = seg_UNet()
segmenter_exp.load_state_dict(seg_data_exp)
segmenter_gaussian = seg_UNet()
segmenter_gaussian.load_state_dict(seg_data_gaussian)
with open('benchmark_100_3.pkl', 'rb') as f:
    data = pickle.load(f)
print("i")
image_data = data["benchmark_sets"]

for i, group in enumerate(image_data):
    #if i == 31 or i == 55 or i == 67 or i == 94: #for 100_2 and previous? #203,229,388 
    if i ==186 or i == 203 or i == 229 or i == 249 or i == 325 or i == 388 or i == 463 or i == 475 or i == 497:
        #plot image
        plt.figure(figsize=(5,5))
        plt.imshow(group[0], cmap='gray',origin="lower")
        plt.axis('off')
        plt.axis('equal')
        plt.show()
        continue
    print(i)
    unet_positions = []
    unet_center_of_mass_positions = []
    unet_gaussian_center_of_mass_positions = []
    unet_gaussian_positions = []
    atomaps_positions = []

    positions_gt = data["dataframes"][i][["x","y"]].to_numpy()*128
    positions_gt = positions_gt[:,::-1]
    labels_gt = data["dataframes"][i]["label"].to_numpy()
    tot_columns = 0
    diff = []
    diff_gaussian = []
    fps_exp = []
    fps_gaussian = []
    fns_exp = []
    fns_gaussian = []
    confusion_matrices_3x3_intermediate = []
    confusion_matrices_gauss_intermediate = []

    tot_correct_exp = []
    tot_correct_gauss = []
    tot_num_positions = []

    for j, image in enumerate(group):
        positions_gt = data["dataframes"][i][["x","y"]].to_numpy()*128
        positions_gt = positions_gt[:,::-1]
        labels_gt = data["dataframes"][i]["label"].to_numpy()

        raw_image = image
        normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
        normalized_image = normalized_image[np.newaxis, :, :]
        image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        localizer = localizer.cuda()
        # Pass the image tensor through the model
        localizer.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            predicted_output = localizer(image_tensor)
        end_time = time.perf_counter()
        times_through_model_3x3.append(end_time-start_time)
        predicted_localization = postprocess_output(predicted_output)
        predicted_localization_save = predicted_localization.copy()
        prediction = torch.tensor(np.where(predicted_localization >= 0.10, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
        prediction_for_com = prediction.squeeze(0).squeeze(0).cpu().numpy()

        localizer_gaussian_gt = localizer_gaussian_gt.cuda()       
        localizer_gaussian_gt.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            predicted_output_gaussian = localizer_gaussian_gt(image_tensor)
        end_time = time.perf_counter()
        times_through_model_gauss.append(end_time-start_time)
        
        predicted_localization_gaussian = postprocess_output(predicted_output_gaussian)
        predicted_localization_save_gaussian = predicted_localization_gaussian.copy()
        prediction_gaussian_save = torch.tensor(np.where(predicted_localization_gaussian >= 0.1, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)
        #plt.figure()
        #plt.imshow(predicted_localization_save_gaussian, cmap='gray', origin="lower")
        #plt.axis('off')
        #plt.show()
        #use region props to find all the centroids of the bright regions in prediction_gaussian
        #import regionprops and label
        labeled_image = label(prediction_for_com)
        regions = regionprops(labeled_image)
        centers_of_mass = []
        for region in regions:
            # Create a binary mask for this region
            mask = (labeled_image == region.label)
            exp_areas.append(region.area)
            # Extract the region's pixel values from the grayscale image
            region_values = mask * predicted_localization_save
            
            # Compute the center of mass for the region using the grayscale values
            center = center_of_mass(region_values)
            centers_of_mass.append(center)

        prediction_gaussian = prediction_gaussian_save.squeeze(0).squeeze(0).cpu().numpy()
        
        start_time = time.perf_counter()
        labeled_image = label(prediction_gaussian)
        regions = regionprops(labeled_image)
        centroids = [region.centroid for region in regions]
        centroids = np.array(centroids)
        centers_of_mass_gaussian = []
        for region in regions:
            # Create a binary mask for this region
            mask = (labeled_image == region.label)
            
            # Extract the region's pixel values from the grayscale image
            region_values = mask * predicted_localization_save_gaussian
            com_areas.append(region.area)
            # Compute the center of mass for the region using the grayscale values
            center = center_of_mass(region_values)
            centers_of_mass_gaussian.append(center)
        end_time = time.perf_counter()
        times_gaussian_com.append(end_time-start_time)
        
        #plt.figure()
        #plt.imshow(labeled_image, cmap=plt.cm.nipy_spectral, origin='lower')
        #plt.scatter(centroids[:, 1], centroids[:, 0], c='r', marker='x')  # note the swapping of x and y here
        #plt.axis('off')
        #plt.colorbar()  # Optional, but it gives a color reference for label values.
        #plt.show()

    

        def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
            (x, y) = xdata_tuple
            xo = float(xo)
            yo = float(yo)    
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
            g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
            return g.ravel()
        
        start_time = time.perf_counter()
        subpixel_centroids = []
        patch_size = 3 # only odd numbers please
        for region in regions:
            y0, x0 = region.centroid
            y0, x0 = int(y0), int(x0)
            # Ensure the patch stays within the image bounds
            y_start = max(0, y0 - patch_size)
            y_end = min(prediction_gaussian.shape[0], y0 + patch_size+2)
            x_start = max(0, x0 - patch_size)
            x_end = min(prediction_gaussian.shape[1], x0 + patch_size+2)
            
            patch = predicted_localization_save_gaussian[y_start:y_end, x_start:x_end]
            #plt.imshow(patch, cmap='gray', origin="lower")
            #plt.show()
            x = np.linspace(0, patch.shape[1]-1, patch.shape[1])
            y = np.linspace(0, patch.shape[0]-1, patch.shape[0])
            x, y = np.meshgrid(x, y)

            # Initial guess for Gaussian parameters:
            initial_params = (patch.max(), patch_size, patch_size, 1, 1, 0, patch.min())
            
            try:
                popt, _ = curve_fit(twoD_Gaussian, (x, y), patch.ravel(), p0=initial_params)
                subpixel_centroids.append((y0-patch_size+popt[2], x0-patch_size+popt[1]))
            except:
                # If fitting fails for some reason, use original centroid
                subpixel_centroids.append((y0, x0))
        pred_gaussian_positions = np.array(subpixel_centroids)
        end_time = time.perf_counter()
        times_gaussian_fit.append(end_time-start_time)

        analyzer = Analyzer()
        start_time = time.perf_counter()
        pred_positions = analyzer.return_positions_experimental(predicted_localization)+[0.5,0.5]
        end_time = time.perf_counter()
        times_exp.append(end_time-start_time)
        if j == 0:
            tot_columns = len(pred_positions)
        #ignore points close to edge

        pred_gaussian_positions = pred_gaussian_positions[
            np.logical_and.reduce((
                pred_gaussian_positions[:,0] > 5, 
                pred_gaussian_positions[:,0] < 123, 
                pred_gaussian_positions[:,1] > 5, 
                pred_gaussian_positions[:,1] < 123
            ))
        ]

        pred_positions = pred_positions[
            np.logical_and.reduce((
                pred_positions[:,0] > 5, 
                pred_positions[:,0] < 123, 
                pred_positions[:,1] > 5, 
                pred_positions[:,1] < 123
            ))
        ]
        position_criteria_mask = np.logical_and.reduce((
            positions_gt[:,0] > 5, 
            positions_gt[:,0] < 123, 
            positions_gt[:,1] > 5, 
            positions_gt[:,1] < 123,
        ))

        filtered_positions = positions_gt[position_criteria_mask]
        filtered_labels = labels_gt[position_criteria_mask].astype(int)

        pos_gt_gaussian = positions_gt.copy()
        pos_gt_3x3 = positions_gt.copy()

        positions_gt = positions_gt[
            np.logical_and.reduce((
                positions_gt[:,0] > 5, 
                positions_gt[:,0] < 123, 
                positions_gt[:,1] > 5, 
                positions_gt[:,1] < 123
            ))
        ]
        diffs,fps,fns = closest_pairs(pred_positions,positions_gt)
        fps_exp.append(len(fps))
        fns_exp.append(len(fns))
        diff.append(len(diffs))
        unet_positions.append(pred_positions)
        unet_center_of_mass_positions.append(centers_of_mass)

        diffs_gaussian,fps_gauss,fns_gauss = closest_pairs(pred_gaussian_positions,positions_gt)
        fps_gaussian.append(len(fps_gauss))
        fns_gaussian.append(len(fns_gauss))
        diff_gaussian.append(len(diffs_gaussian))
        unet_gaussian_positions.append(pred_gaussian_positions)
        unet_gaussian_center_of_mass_positions.append(centers_of_mass_gaussian)
        
        positions_atomaps = pred_positions.copy()
        atomaps_positions.append(positions_atomaps)

        segmenter_exp = segmenter.cuda()
        segmenter_exp.eval()
        #plt.figure()
        #plt.imshow(prediction_gaussian, cmap='gray', origin="lower")
        #plt.show()
        with torch.no_grad():
            #print(image_tensor.shape,predicted_output.shape,density_map.shape)
            #predicted_output = segmenter(image_tensor, predicted_output,density_map.to(device),ld_map.to(device))
            predicted_output_exp = segmenter_exp(prediction.to(device))
        predicted_segmentation = postprocess_output(predicted_output_exp)
        binary_segmentation_exp = np.where(predicted_segmentation > 0.5, 1, 0)

        segmenter_gaussian = segmenter_gaussian.cuda()
        segmenter_gaussian.eval()
        with torch.no_grad():
            predicted_output_gaussian = segmenter_gaussian(prediction_gaussian_save.to(device))
        predicted_segmentation_gaussian = postprocess_output(predicted_output_gaussian)
        binary_segmentation_gaussian = np.where(predicted_segmentation_gaussian > 0.5, 1, 0)
        #filtered_positions = np.flip(filtered_positions,1)
        #mask_pos = ~np.all(np.isclose(positions_gt, 129.0), axis=-1)
        #print(labels_gt)
        #mask_lab = ~np.isclose(labels_gt, 129.0)
        
        #masked_positions = positions_gt[mask_pos] - 0.5
        #masked_labels = labels_gt[mask_lab]
        #print(masked_positions.shape,masked_labels.shape)
        #if i==0 and epoch == 6:
        #    plt.imshow(predicted_regions[i].cpu().numpy().squeeze())
        #    plt.scatter(masked_positions[:,1],masked_positions[:,0])
        #    plt.show()
        
        def filter_positions_labels(positions, labels, binary_segmentation):
            mask = [binary_segmentation[int(pos[0]), int(pos[1])] == 1 for pos in positions]
            filtered_positions = positions[mask]
            filtered_labels = labels[mask]
            return filtered_positions, filtered_labels

        # Usage example
        filtered_positions_3x3, filtered_labels_3x3 = filter_positions_labels(filtered_positions, filtered_labels, prediction_for_com)
        filtered_positions_gaussian, filtered_labels_gaussian = filter_positions_labels(filtered_positions, filtered_labels, prediction_gaussian)
        #print(filtered_positions_3x3.shape,filtered_positions_gaussian.shape)
        #print(filtered_labels_3x3.shape,filtered_labels_gaussian.shape)
        pred_labels = label_points(filtered_positions_3x3, binary_segmentation_exp)
        #plt.subplot(1,2,1)
        #plt.imshow(binary_segmentation_exp, cmap='gray', origin="lower")
        #plt.scatter(filtered_positions[:,1],filtered_positions[:,0], s=5,c=pred_labels)
        #plt.show()

        #print(len(filtered_labels),len(pred_labels))
        #print(filtered_labels,pred_labels)
        #print("Unique values in true labels:", np.unique(filtered_labels))
        #print("Unique values in predicted labels:", np.unique(pred_labels))
        total_correct_3x3 = accuracy_score(filtered_labels_3x3, pred_labels, normalize=False)
        total_predictions = len(filtered_positions_3x3)

        pred_labels_gaussian = label_points(filtered_positions_gaussian, binary_segmentation_gaussian)
        total_correct_gaussian = accuracy_score(filtered_labels_gaussian, pred_labels_gaussian, normalize=False)

        #if conf_mat_3x3.shape != (2,2):
        #    print("HELLO")
        #    print("conf_mat_3x3",conf_mat_3x3.shape)
        #    print("filtered_labels_3x3",filtered_labels_3x3.shape)
        #    print("pred_labels",pred_labels.shape)
        #    print("filtered_positions_3x3",filtered_positions_3x3.shape)
        #if conf_mat_gaussian.shape != (2,2):
        #    print("GAUSS HLELLo")
        #    print("conf_mat_gaussian",conf_mat_gaussian.shape)
        #    print("filtered_labels_gaussian",filtered_labels_gaussian.shape)
        #    print("pred_labels_gaussian",pred_labels_gaussian.shape)
        #    print("filtered_positions_gaussian",filtered_positions_gaussian.shape)


        confusion_matrices_3x3_intermediate.append(confusion_matrix(filtered_labels_3x3, pred_labels, labels=[1,0]))
        confusion_matrices_gauss_intermediate.append(confusion_matrix(filtered_labels_gaussian, pred_labels_gaussian, labels=[1,0]))


        tot_correct_exp.append(total_correct_3x3)
        tot_correct_gauss.append(total_correct_gaussian)
        tot_num_positions.append(total_predictions)
        #print(i,j,total_correct_3x3/total_predictions,total_correct_gaussian/total_predictions)    
        #plot the image, predicted positions colored by predicted label
        #plt.figure(figsize=(5,5))
        #plt.imshow(raw_image, cmap='gray', origin="lower")
        #plt.scatter(filtered_positions[:,1],filtered_positions[:,0], s=5,c=pred_labels_gaussian)
        #plt.axis('off')
        #plt.show()

        #accuracy_exp.append(total_correct_train/total_predictions_train
        #if j == 0:
            #plt.figure(figsize=(15,5))
            #plt.subplot(1,3,1)
            #plt.imshow(image, cmap='gray', origin="lower")
            #plt.subplot(1,3,2)
            #plt.imshow(predicted_localization_gaussian, cmap='gray', origin="lower")
            #plt.scatter(pred_gaussian_positions[:,1],pred_gaussian_positions[:,0], s=5,c='r')
            #plt.subplot(1,3,3)
            #plt.imshow(predicted_localization_save, cmap='gray', origin="lower")
            #plt.scatter(pred_positions[:,1]-0.5,pred_positions[:,0]-0.5, s=5,c='r')
            #plt.show()
    differences.append(diff)
    differences_gaussian.append(diff_gaussian)
    fp_exp.append(fps_exp)
    fp_gaussian.append(fps_gaussian)
    fn_exp.append(fns_exp)
    fn_gaussian.append(fns_gaussian)
    correct_predictions_3x3.append(tot_correct_exp)
    correct_predictions_gauss.append(tot_correct_gauss)
    tot_predictions.append(tot_num_positions)  
    confusion_matrices_3x3.append(confusion_matrices_3x3_intermediate)
    confusion_matrices_gauss.append(confusion_matrices_gauss_intermediate) 

    unet_errors = []
    unet_com_errors = []
    atomaps_errors = []
    unet_gaussian_errors = []
    unet_gaussian_com_errors = []
    for positions_unet,positions_unet_gaussian,positions_atomaps,positions_unet_com, positions_unet_gaussian_com in zip(unet_positions,unet_gaussian_positions,atomaps_positions, unet_center_of_mass_positions, unet_gaussian_center_of_mass_positions):
        nn_unet = analyzer.assign_nearest(positions_unet,positions_gt,3)
        nn_atomaps = analyzer.assign_nearest(positions_atomaps,positions_gt,3)
        nn_unet_gaussian = analyzer.assign_nearest(positions_unet_gaussian,positions_gt,3)
        nn_unet_com = analyzer.assign_nearest(positions_unet_com,positions_gt,3)
        nn_unet_gaussian_com = analyzer.assign_nearest(positions_unet_gaussian_com,positions_gt,3)
        unet_error = analyzer.calculate_error(positions_unet,nn_unet)
        atomaps_error = analyzer.calculate_error(positions_atomaps,nn_atomaps)
        unet_gaussian_error = analyzer.calculate_error(positions_unet_gaussian,nn_unet_gaussian)
        unet_com_error = analyzer.calculate_error(positions_unet_com,nn_unet_com)
        unet_gaussian_com_error = analyzer.calculate_error(positions_unet_gaussian_com,nn_unet_gaussian_com)
        unet_errors.append(unet_error)
        atomaps_errors.append(atomaps_error)
        unet_gaussian_errors.append(unet_gaussian_error)
        unet_com_errors.append(unet_com_error)
        unet_gaussian_com_errors.append(unet_gaussian_com_error)


    #print(unet_errors)
    mean_values_u = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
    mean_values_u_gaussian = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_gaussian_errors])
    mean_values_u_com = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_com_errors])
    mean_values_u_gaussian_com = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_gaussian_com_errors])
    #print(mean_values_u)
    standard_deviations_u = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_errors])
    standard_deviations_u_gaussian = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_gaussian_errors])
    standard_deviations_u_com = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_com_errors])
    standard_deviations_u_gaussian_com = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in unet_gaussian_com_errors])

    #print(mean_values_u,standard_deviations_u)
    #mean_values_a = np.array([statistics.mean(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
    #standard_deviations_a = np.array([statistics.stdev(np.array(sublist)[np.isfinite(sublist)]) for sublist in atomaps_errors])
    ssim = data["benchmark_ssims"][i]
    #if i == 9:
        #remove the last two elements of ssim (bug)
    #    ssim = ssim[:-2]
    #print(ssim)
    means_u.append(mean_values_u)  
    stds_u.append(standard_deviations_u)
    means_u_gaussian.append(mean_values_u_gaussian)
    stds_u_gaussian.append(standard_deviations_u_gaussian)
    means_u_com.append(mean_values_u_com)
    stds_u_com.append(standard_deviations_u_com)
    means_u_gaussian_com.append(mean_values_u_gaussian_com)
    stds_u_gaussian_com.append(standard_deviations_u_gaussian_com)
    ssims.append(ssim)

print(len(means_u))

#print mean of times
print("gauss fit",np.mean(times_gaussian_fit))
print("gauss com",np.mean(times_gaussian_com))
print("exponential",np.mean(times_exp))
print("mean com area",np.mean(com_areas))
print("mean exp area",np.mean(exp_areas))
print("mean false positives exp",np.mean(fp_exp))
print("mean false positives gauss",np.mean(fp_gaussian))
print("mean false negatives exp",np.mean(fn_exp))
print("mean false negatives gauss",np.mean(fn_gaussian))
print("mean times through model 3x3",np.mean(times_through_model_3x3))
print("mean times through model gauss",np.mean(times_through_model_gauss))

#%%
accuracy_3x3 = np.sum(correct_predictions_3x3,axis=0)/np.sum(tot_predictions,axis=0)
accuracy_gaussian = np.sum(correct_predictions_gauss,axis=0)/np.sum(tot_predictions,axis=0)
print(accuracy_3x3)
print(accuracy_gaussian)
acc_3x3_flip = np.flip(accuracy_3x3)
acc_gaussian_flip = np.flip(accuracy_gaussian)
print(acc_3x3_flip)
print(acc_gaussian_flip)
print(confusion_matrices_3x3)
#print(confusion_matrices_3x3)
confusion_array = np.array(confusion_matrices_3x3) 
confusion_array_gauss = np.array(confusion_matrices_gauss)
summed_confusion_matrices = confusion_array.sum(axis=0)
summed_confusion_matrices_gauss = confusion_array_gauss.sum(axis=0)
#print(summed_confusion_matrices)
#print(summed_confusion_matrices_gauss)

def percentage_confusion_matrix(confusion_array):
    percentage_confusion_matrices = np.empty(confusion_array.shape)
    for i in range(confusion_array.shape[0]):  # iterating over each row (image)
        for j in range(confusion_array.shape[1]):  # iterating over each column (noise level)
            # Calculate the total count for a single confusion matrix.
            total = confusion_array[i, j].sum()
            
            # Convert each element to percentage.
            percentage_confusion_matrices[i, j] = (confusion_array[i, j] / total) * 100
    return percentage_confusion_matrices

mean_percentage_confusion_matrix = np.mean(percentage_confusion_matrix(confusion_array),axis=0)
mean_percentage_confusion_matrix_gauss = np.mean(percentage_confusion_matrix(confusion_array_gauss),axis=0)
print(mean_percentage_confusion_matrix)
print(mean_percentage_confusion_matrix_gauss)

def calculate_metrics(confusion_array, metric):
    """
    Calculate the desired metric (precision, recall, or F1 score) for each noise level.

    :param confusion_array: np.array, confusion matrix array (500, 12, 2, 2)
    :param metric: str, type of metric to calculate ('precision', 'recall', or 'f1')
    :return: np.array, mean scores for each noise level
    """
    all_scores = np.zeros((confusion_array.shape[0], confusion_array.shape[1]))  # (500, 12)
    
    for i in range(confusion_array.shape[0]):  # iterating over each row (image)
        for j in range(confusion_array.shape[1]):  # iterating over each column (noise level)
            tn, fp, fn, tp = confusion_array[i, j].ravel()
            
            if metric == 'recall':
                score = tp / (tp + fn) if (tp + fn) > 0 else 0  # to avoid division by zero
            elif metric == 'precision':
                score = tp / (tp + fp) if (tp + fp) > 0 else 0  # to avoid division by zero
            elif metric == 'f1':
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                raise ValueError("Metric not recognized. Use 'precision', 'recall', or 'f1'.")
            
            all_scores[i, j] = score
    
    return all_scores.mean(axis=0)

# Calculate the mean recall for each noise level
mean_recall_3x3 = calculate_metrics(confusion_array, 'recall')
mean_recall_gauss = calculate_metrics(confusion_array_gauss, 'recall')
mean_precision_3x3 = calculate_metrics(confusion_array, 'precision')
mean_precision_gauss = calculate_metrics(confusion_array_gauss, 'precision')
mean_f1_3x3 = calculate_metrics(confusion_array, 'f1')
mean_f1_gauss = calculate_metrics(confusion_array_gauss, 'f1')
print(mean_precision_3x3)
print(mean_precision_gauss)
print(mean_recall_3x3)
print(mean_recall_gauss)

#print(summed_confusion_matrices_gauss)
# %%
#average the means and stds in means_u and stds_u
print("gauss fit",np.mean(times_gaussian_fit))
print("gauss com",np.mean(times_gaussian_com))
print("exponential",np.mean(times_exp))
print("mean com area",np.mean(com_areas))
print("mean exp area",np.mean(exp_areas))
print("mean false positives exp",np.mean(fp_exp))
print("mean false positives gauss",np.mean(fp_gaussian))
print("mean false negatives exp",np.mean(fn_exp))
print("mean false negatives gauss",np.mean(fn_gaussian))
print(np.shape(means_u))
print(np.shape(differences))
# Find the row indices where all elements of differences are zeros
zero_rows_indices = np.array(np.where(np.all(np.array(differences) == 0, axis=1)))[0]
print(zero_rows_indices)
print(np.shape(means_u))
print(type(means_u))
filtered_means_u = np.array(means_u)[zero_rows_indices]
filtered_stds_u = np.array(stds_u)[zero_rows_indices]
filtered_means_u_gaussian_com = np.array(means_u_gaussian_com)[zero_rows_indices]
filtered_stds_u_gaussian_com = np.array(stds_u_gaussian_com)[zero_rows_indices]


mean_values_u = np.sum(means_u,axis=0)/len(means_u)
standard_deviations_u = np.sum(stds_u,axis=0)/len(stds_u)
ssim_values = np.sum(ssims,axis=0)/len(ssims)
from scipy.stats import mode
mean_diff = np.sum(differences,axis=0)/len(differences)
median_diff = np.median(differences,axis=0)
mode_diff, counts = mode(differences, axis=0) 
print(mode_diff, counts)
mean_values_u_gaussian = np.sum(means_u_gaussian,axis=0)/len(means_u_gaussian)
standard_deviations_u_gaussian = np.sum(stds_u_gaussian,axis=0)/len(stds_u_gaussian)
mean_diff_gaussian = np.sum(differences_gaussian,axis=0)/len(differences_gaussian)
median_diff_gaussian = np.median(differences_gaussian,axis=0)
mode_diff_gaussian, counts_gaussian = mode(differences_gaussian, axis=0)
mean_fps_exp = np.sum(fp_exp,axis=0)/len(fp_exp)
mean_fps_gaussian = np.sum(fp_gaussian,axis=0)/len(fp_gaussian)
mean_fns_exp = np.sum(fn_exp,axis=0)/len(fn_exp)
mean_fns_gaussian = np.sum(fn_gaussian,axis=0)/len(fn_gaussian)
print(mode_diff_gaussian, counts_gaussian)
mean_values_u_com = np.sum(means_u_com,axis=0)/len(means_u_com)
standard_deviations_u_com = np.sum(stds_u_com,axis=0)/len(stds_u_com)
mean_values_u_gaussian_com = np.sum(means_u_gaussian_com,axis=0)/len(means_u_gaussian_com)
standard_deviations_u_gaussian_com = np.sum(stds_u_gaussian_com,axis=0)/len(stds_u_gaussian_com)

mean_values_u_filtered = np.sum(filtered_means_u,axis=0)/len(filtered_means_u)
standard_deviations_u_filtered = np.sum(filtered_stds_u,axis=0)/len(filtered_stds_u)
mean_values_u_gaussian_com_filtered = np.sum(filtered_means_u_gaussian_com,axis=0)/len(filtered_means_u_gaussian_com)
standard_deviations_u_gaussian_com_filtered = np.sum(filtered_stds_u_gaussian_com,axis=0)/len(filtered_stds_u_gaussian_com)





fig, ax1 = plt.subplots()

# Plotting data on primary y-axis
ax1.errorbar(ssim_values, mean_values_u, standard_deviations_u, linestyle='-', marker='o', color='darkgreen',capsize=2,capthick=2, markerfacecolor='darkgreen', markeredgecolor='black')
ax1.set_ylabel('Mean error [pixels]', color='k', weight='bold')
ax1.tick_params(axis='y', labelcolor='k')
ax1.tick_params(axis='both', labelsize=11)
from matplotlib.ticker import MultipleLocator

# Set major ticks every 0.2 from 0 to 1.1
ax1.set_yticks(np.arange(0, 1.2, 0.2))

# Set minor ticks every 0.1
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

# Set major tick length larger and display labels, while setting minor tick length smaller and not displaying labels
ax1.tick_params(axis='y', which='major', length=10, labelsize=11)  # Adjust length and label size as needed
ax1.tick_params(axis='y', which='minor', length=5, labelsize=0)
ax1.set_xlabel('SSIM',weight='bold')
#ax1.errorbar(ssim_values, mean_values_u_com, standard_deviations_u_com, linestyle='-', marker='^', color='k',capsize=2,capthick=2, markerfacecolor='darkgreen', markeredgecolor='black')
ax1.errorbar(ssim_values, mean_values_u_gaussian_com, standard_deviations_u_gaussian_com, linestyle='-', marker='o', color='darkred',capsize=2,capthick=2, markerfacecolor='darkred', markeredgecolor='black')
ax1.errorbar(ssim_values, mean_values_u_gaussian, standard_deviations_u_gaussian, linestyle='-', marker='o', color='k',capsize=2,capthick=2, markerfacecolor='k', markeredgecolor='black')
ax1.legend(['3x3 gt','Gaussian gt, CoM','Gaussian gt, Fitting'], facecolor='white', edgecolor='black', loc='upper right',prop={'weight':'bold'}, title='Mean error')
# Create a twin y-axis
ax2 = ax1.twinx()

# Plotting data on secondary y-axis
#ax2.plot(ssim_values, mean_diff, linestyle='-', color='gray', markerfacecolor='gray')
#ax2.plot(ssim_values, mean_diff_gaussian, linestyle='--', color='gray', markerfacecolor='gray')
ax2.plot(ssim_values, mean_fps_exp, linestyle='-', color='gray', markerfacecolor='gray')
ax2.plot(ssim_values, mean_fns_exp, linestyle=':', color='gray', markerfacecolor='gray')
ax2.plot(ssim_values, mean_fps_gaussian, linestyle='--', color='gray', markerfacecolor='gray')
ax2.plot(ssim_values, mean_fns_gaussian, linestyle='-.', color='gray', markerfacecolor='gray')
# Set secondary y-axis color
ax2.set_ylabel('Mean # False predictions', color='gray',weight='bold')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.tick_params(axis='y', labelsize=11)
ax2.set_yticks(np.arange(0, 5, step=1))
#ax2.legend(['Mean 3x3 gt','Mean Gaussian gt'],facecolor='white', edgecolor='black', loc=(0.6,0.55),prop={'weight':'bold'},title='False predictions')
ax2.legend(['FP 3x3 gt','FN 3x3 gt','FP Gaussian gt','FN Gaussian gt'],facecolor='white', edgecolor='black', loc=(0.646,0.43),prop={'weight':'bold'},title='False predictions')
plt.title('Benchmark (Avg. 500 Images)',weight='bold')
plt.show()

#plot acc_3x3_flip and acc_gaussian_flip with ssim_values on x-axis as a line plot
fig, ax1 = plt.subplots()
ax1.plot(ssim_values, accuracy_3x3, linestyle='-', marker='o', color='darkgreen', markerfacecolor='darkgreen', markeredgecolor='black')
ax1.plot(ssim_values, accuracy_gaussian, linestyle='-', marker='o', color='darkred', markerfacecolor='darkred', markeredgecolor='black')
ax1.set_ylim([0.88,1.01])
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xlabel('SSIM',weight='bold')
ax1.set_yticks(np.arange(0.9, 1.04, step=0.05))
ax1.yaxis.set_minor_locator(MultipleLocator(0.025))
ax1.tick_params(axis='y', which='major', length=10, labelsize=11)  # Adjust length and label size as needed
ax1.tick_params(axis='y', which='minor', length=5, labelsize=0)
ax2= ax1.twinx()
ax2.plot(ssim_values, mean_fps_gaussian, linestyle='--', color='gray', markerfacecolor='gray')
ax2.plot(ssim_values, mean_fps_exp, linestyle='-', color='gray', markerfacecolor='gray')
plt.xlabel('SSIM',weight='bold')
ax1.set_ylabel('Accuracy',weight='bold')
ax2.set_ylabel('Mean # False positives', color='gray',weight='bold')
ax2.set_yticks(np.arange(0, 5, step=1))
ax2.tick_params(axis='y', labelcolor='gray')
ax2.tick_params(axis='y', labelsize=11)
ax1.legend(['3x3 gt','Gaussian gt'], facecolor='white', edgecolor='black', loc=(0.7,0.25),prop={'weight':'bold'},title='Accuracy')
ax2.legend(['Gaussian gt','3x3 gt'],facecolor='white', edgecolor='black', loc=(0.7,0.05),prop={'weight':'bold'},title='False positives')
plt.title('Benchmark (Avg. 500 Images)')
plt.show()

plt.figure()
plt.plot(ssim_values, mean_recall_3x3, linestyle='-', marker='o', color='darkgreen', markerfacecolor='darkgreen', markeredgecolor='black')
plt.plot(ssim_values, mean_recall_gauss, linestyle='-', marker='o', color='darkred', markerfacecolor='darkred', markeredgecolor='black')
#plt.ylim([0.65,1.05])
minor_locator = MultipleLocator(0.025)  # Setting the interval for minor ticks
plt.gca().yaxis.set_minor_locator(minor_locator)
plt.yticks(np.arange(0.7, 1.0, step=0.1))
plt.tick_params(axis='y', which='major', length=10, labelsize=11)  # Adjust length and label size as needed
plt.tick_params(axis='y', which='minor', length=5, labelsize=0)
plt.tick_params(axis='both', labelsize=11)
plt.xlabel('SSIM',weight='bold')
plt.ylabel('Recall',weight='bold')
plt.legend(['3x3 gt','Gaussian gt'], facecolor='white', edgecolor='black', loc='lower right',prop={'weight':'bold'},title='Recall')
plt.title('Recall (Avg. 500 Images)')
plt.show()

plt.figure()
plt.plot(ssim_values, mean_precision_3x3, linestyle='-', marker='o', color='darkgreen', markerfacecolor='darkgreen', markeredgecolor='black')
plt.plot(ssim_values, mean_precision_gauss, linestyle='-', marker='o', color='darkred', markerfacecolor='darkred', markeredgecolor='black')
#plt.ylim([0.65,1.05])
minor_locator = MultipleLocator(0.025)  # Setting the interval for minor ticks
plt.gca().yaxis.set_minor_locator(minor_locator)
plt.yticks(np.arange(0.7, 1.0, step=0.1))
plt.tick_params(axis='y', which='major', length=10, labelsize=11)  # Adjust length and label size as needed
plt.tick_params(axis='y', which='minor', length=5, labelsize=0)
plt.tick_params(axis='both', labelsize=11)
plt.xlabel('SSIM',weight='bold')
plt.ylabel('Precision',weight='bold')
plt.legend(['3x3 gt','Gaussian gt'], facecolor='white', edgecolor='black', loc='lower right',prop={'weight':'bold'},title='Precision')
plt.title('Precision (Avg. 500 Images)')
plt.show()

plt.figure()
plt.plot(ssim_values, mean_f1_3x3, linestyle='-', marker='o', color='darkgreen', markerfacecolor='darkgreen', markeredgecolor='black')
plt.plot(ssim_values, mean_f1_gauss, linestyle='-', marker='o', color='darkred', markerfacecolor='darkred', markeredgecolor='black')
#plt.ylim([0.65,1.05])
minor_locator = MultipleLocator(0.0125)  # Setting the interval for minor ticks
plt.gca().yaxis.set_minor_locator(minor_locator)
plt.yticks([0.95,0.975,1.0])
plt.tick_params(axis='y', which='major', length=10, labelsize=11)  # Adjust length and label size as needed
plt.tick_params(axis='y', which='minor', length=5, labelsize=0)
plt.tick_params(axis='both', labelsize=11)
plt.xlabel('SSIM',weight='bold')
plt.ylabel('F1 score',weight='bold')
plt.legend(['3x3 gt','Gaussian gt'], facecolor='white', edgecolor='black', loc='lower right',prop={'weight':'bold'},title='F1 score')
plt.title('F1 score (Avg. 500 Images)')
plt.show()




print(mean_values_u)
print(mean_values_u_gaussian)
print(mean_values_u_gaussian_com)
print(mean_values_u_com)
print(mean_fps_exp+mean_fns_exp,mean_fps_gaussian+mean_fns_gaussian)
print(mean_f1_3x3,mean_f1_gauss)
#%%
for i in range(len(means_u)):
    plt.figure()
    #if i == 9:
    #    for idx, item in enumerate(ssims[i]):
    #        print(idx, item)
    print(i,len(ssims[i]),len(means_u[i]),len(stds_u[i]))
    plt.errorbar(ssims[i], means_u[i], stds_u[i], linestyle='None', marker='^')
    #plt.errorbar(ssims[i], mean_a[i], stds_a[i], linestyle='None', marker='^')
    plt.xlabel('SSIM')
    plt.ylabel('Mean error')
    plt.legend(['UNET', 'Atomap'])
    plt.show()
# %%
#Try on 5 experimental images
import tifffile as tif
from tifffile import imsave
from PI_U_Net import UNet
from Analyzer import Analyzer
predicted_localizations_list = []
#image_paths = ["data/experimental_data/32bit/torben.tif","data/experimental_data/32bit/larger.tif","data/experimental_data/32bit/very_noisy.tif","data/experimental_data/32bit/other_microscope.tif","data/experimental_data/32bit/grain_boundary.tif","data/experimental_data/32bit/small_pixel_distance.tif"]
#image_paths is all files starting with "particle_dynamic"
image_paths_2 = [f"data/experimental_data/32bit/particle_dynamics{i:03}.tif" for i in range(50)]
data_endings = ["005","006","010","011","014","015","016","018","019","020","024","025","028","035","037","041","042","043","044","048"]
image_paths_3 = []
for data_ending in data_endings:
    image_paths_3.append(f"data/experimental_data/32bit/particle_dynamics{data_ending}.tif")
image_paths_debug = ["data/experimental_data/32bit/particle_dynamics006.tif"]
print(image_paths_3)
image_paths = ["data/experimental_data/32bit/close_columns.tif"]
point_sets = []
images_list = []
for image_path in image_paths_3:
    #seg_data = torch.load("magiskt_bra.pth")
    checkpoint = torch.load('best_test_acc_gaussian_2.pth')   
    seg_data = checkpoint['model_state_dict']
    im_idx = 1
    segmenter = seg_UNet()
    localizer = UNet()
    loc_data = torch.load("best_model_data.pth")
    loaded_model_state_dict = loc_data['model_state_dict']
    localizer.load_state_dict(loaded_model_state_dict)
    segmenter.load_state_dict(seg_data)

    #image_path = "data/experimental_data/32bit/larger.tif"
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    localizer = localizer.cuda()
    # Pass the image tensor through the model
    localizer.eval()
    with torch.no_grad():
        predicted_output = localizer(image_tensor)
    predicted_localization = postprocess_output(predicted_output)
    predicted_localization_save = predicted_localization.copy()

    #if image size is 128x128
    analyzer = Analyzer()
    pred_positions = analyzer.return_positions_experimental(predicted_localization)
    #set all values larger than 0.1 in predicted_localization to 1
    #predicted_localization[predicted_localization > 0.05] = 1
    predicted_localization= np.where((predicted_localization > 0.12),1,0)
    
    localizer_gaussian = UNet()
    loc_data_gaussian = torch.load("model_data_epoch_gaussian72.pth")
    loaded_model_state_dict_gauss = loc_data_gaussian['model_state_dict']
    localizer_gaussian.load_state_dict(loaded_model_state_dict_gauss)
    localizer_gaussian = localizer_gaussian.cuda()
    # Pass the image tensor through the model
    localizer_gaussian.eval()
    with torch.no_grad():
        predicted_output_gaussian = localizer_gaussian(image_tensor)
    predicted_localization_gaussian = postprocess_output(predicted_output_gaussian)
    prediction = torch.tensor(np.where(predicted_localization >= 0.12, 1, 0)[np.newaxis,np.newaxis, :, :],dtype=torch.float32)

    segmenter = segmenter.cuda()
    segmenter.eval()
    with torch.no_grad():
        #print(image_tensor.shape,predicted_output.shape,density_map.shape)
        #predicted_output = segmenter(image_tensor, predicted_output,density_map.to(device),ld_map.to(device))
        predicted_output = segmenter(prediction.to(device))

    predicted_segmentation = postprocess_output(predicted_output)

    # Convert predicted segmentation to binary using threshold of 0.5
    binary_segmentation = np.where(predicted_segmentation > 0.5, 1, 0)
    
    #mask pred_posions with binary_segmentation and return a vector with 1 or 0 depending on if the position is in a bright region or not
    colors = []
    for idx, pos in enumerate(pred_positions):
        x, y = int(pos[1]+0.5), int(pos[0]+0.5)  # Get the x and y positions
        if idx == 12:
            print(pos)
            print(x,y)
            print(binary_segmentation[y, x])
            #plt.figure()
            #plt.imshow(binary_segmentation,interpolation='none', cmap='gray', origin="upper", aspect='equal')
            #highlight pixel x,y
            #plt.scatter(x,y,s=40,c='b')
            #plt.axis('off')
            #plt.show()
        if binary_segmentation[y, x] == 1:
            colors.append('springgreen')
        else:
            colors.append('darkorange')

    plt.figure()
    #plt.subplot(1,4,1)
    #plt.imshow(image_tensor.cpu().squeeze(0).squeeze(0).numpy(), interpolation='none', cmap='gray', origin="upper", aspect='equal')
    #plt.scatter(pred_positions[:,1], pred_positions[:,0], s=10, c=colors)
    #plt.axis('off')
    #plt.subplot(1,4,2)
    plt.imshow(predicted_localization_save,interpolation='none', cmap='gray', origin="upper", aspect='equal')
    plt.axis('off')
    plt.show()
    #plt.imshow(predicted_localization_gaussian,interpolation='none', cmap='gray', origin="upper", aspect='equal')
    #plt.axis('off')
    #plt.subplot(1,4,4)
    plt.figure()
    plt.imshow(binary_segmentation,interpolation='none', cmap='gray', origin="upper", aspect='equal')
    plt.scatter(pred_positions[:,1], pred_positions[:,0], s=10, c=colors)
    plt.axis('off')
    plt.show()
    
    #plt.figure(figsize=(5,5))
    #plt.imshow(image_tensor.cpu().squeeze(0).squeeze(0).numpy(),interpolation='none', cmap='gray', origin="upper", aspect='equal')
    #plt.scatter(pred_positions[:,1], pred_positions[:,0], s=40, c=colors)
    #plt.axis('off')
    #plt.show()
    id = 12
    plt.figure(figsize=(10,20))
    plt.subplot(1,2,1)
    plt.imshow(image_tensor.cpu().squeeze(0).squeeze(0).numpy(),interpolation='none', cmap='gray', origin="upper", aspect='equal')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(image_tensor.cpu().squeeze(0).squeeze(0).numpy(),interpolation='none', cmap='gray', origin="upper", aspect='equal')
    plt.scatter(pred_positions[:,1], pred_positions[:,0], s=20, c=colors)
    #plt.scatter(pred_positions[id,1], pred_positions[id,0], s=40, c='b')
    plt.axis('off')
    plt.show()
    #interpolation = None
    plt.figure(figsize=(5,5))
    plt.imshow(image_tensor.cpu().squeeze(0).squeeze(0).numpy(),interpolation='none', cmap='gray', origin="upper", aspect='equal')
    plt.axis('off')
    plt.show()
    plt.figure(figsize=(5,5))
    plt.imshow(image_tensor.cpu().squeeze(0).squeeze(0).numpy(),interpolation='none', cmap='gray', origin="upper", aspect='equal')
    plt.scatter(pred_positions[:,1]+0.5, pred_positions[:,0]+0.5, s=5, c=colors)
    plt.axis('off')
    plt.show()

    point_sets.append(pred_positions)
    fig, ax = plt.subplots(figsize=(15,15))
    ax.imshow(image_tensor.cpu().squeeze(0).squeeze(0).numpy(), interpolation='none', cmap='gray', origin="upper", aspect='equal')
    ax.scatter(pred_positions[:,1]+0.5, pred_positions[:,0]+0.5, s=120, c=colors)
    ax.axis('off')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images_list.append(data)

    # Close the current figure to release memory
    plt.close(fig)

# Convert list of images to a numpy array
images_array = np.array(images_list)
imsave('stacked_images.tif', images_array)

plt.figure()
plt.axis("off")
plt.axis('equal')
for point_set in point_sets:
    plt.scatter(point_set[:,1],point_set[:,0],s=10,c='k')
    #set origin lower
plt.show()
#%%
def calculate_msd_and_error(point_sets,epsilon=0.4,pixel_size = 25.357):
    """
    Calculate the mean position, MSD, and corrected delta MSD for linked points with variable numbers across frames,
    considering the correct error propagation for each displacement.

    Parameters:
    - point_sets: A list of arrays, each containing a set of points (x, y) for each frame.
    - sigma: The standard deviation of positional error.

    Returns:
    - mean_positions: Mean position of each list of linked positions.
    - msds: Mean squared displacements of each list of linked positions.
    - delta_msds: Corrected errors in the mean squared displacements.
    """

    # Initialize list to store the linked positions across frames
    linked_positions = []
    pixel_size = 25.357
    # Link positions from the first frame to the closest in the subsequent frames
    for point in point_sets[0]:
        linked_position = [point]  # Start with the initial position
        for subsequent_point_set in point_sets[1:]:
            # Compute distances to all points in the next frame
            distances = np.linalg.norm(subsequent_point_set - point, axis=1)
            # Find the closest point
            closest_idx = np.argmin(distances)
            closest_point = subsequent_point_set[closest_idx]
            # Update the point for the next iteration
            point = closest_point
            # Add the closest point to the linked positions
            linked_position.append(closest_point)
        linked_positions.append(linked_position)

    # Convert to numpy array for easier manipulation
    linked_positions = np.array(linked_positions, dtype=object)

    # Initialize arrays to store results
    mean_positions = np.zeros((len(linked_positions), 2))
    msds = np.zeros(len(linked_positions))
    delta_msds = np.zeros(len(linked_positions))

    # Calculate mean position, MSD, and corrected delta MSD for each list of linked positions
    for i, positions in enumerate(linked_positions):
        positions = np.array(positions)
        mean_position = positions.mean(axis=0)
        displacements = (positions - mean_position)
        squared_displacements = np.sum(displacements**2, axis=1)

        mean_positions[i] = mean_position
        msds[i] = squared_displacements.mean()*pixel_size**2

        # Ensure squared_displacements is a 1D numpy array of floats
        squared_displacements = np.array(squared_displacements, dtype=float).flatten()
        displacements = np.sqrt(squared_displacements)  # get the displacement from squared displacement
        
        error_terms = [(2*np.sqrt(2)*epsilon*di*pixel_size*pixel_size) for di in displacements]
        msd_error = (1/len(squared_displacements)) * np.sqrt(np.sum(np.array(error_terms)**2))
        delta_msds[i] = msd_error

    return mean_positions, msds, delta_msds
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Call the function with the example point sets
mean_positions, msds, delta_msds = calculate_msd_and_error(point_sets)
#print(mean_positions, msds, delta_msds)
big = msds + delta_msds
small = msds - delta_msds
point_sizes = msds / np.max(msds) *1800  # Normalize and scale
point_sizes_big = big / np.max(msds) *1800  # Normalize and scale
point_sizes_small = small / np.max(msds) *1800  # Normalize and scale
print(point_sizes,point_sizes_big,point_sizes_small)
# Normalize msds for color mapping
norm = plt.Normalize(vmin=np.min(msds), vmax=np.max(msds))

# Create a colormap
cmap = plt.get_cmap('coolwarm')

plt.figure(figsize=(15, 15))

# Plot larger red circles (MSD + error)
colors_big = cmap(norm(msds + delta_msds))
plt.scatter(mean_positions[:, 1], mean_positions[:, 0], facecolors='none', edgecolors=colors_big, linestyle='-', linewidths=3, s=point_sizes_big)

# Plot filled circles (actual MSD)
plt.scatter(mean_positions[:, 1], mean_positions[:, 0], s=point_sizes, c=msds, cmap='coolwarm', norm=norm)

# Create a colorbar
cbar = plt.colorbar(shrink=0.55)
cbar.ax.set_ylabel('MSD [pm$^2$]', weight='bold', fontsize=20)
cbar.ax.tick_params(labelsize=20)
#set cbar tick bold text
cbar.ax.yaxis.set_tick_params(width=2)
# Plot smaller blue circles (MSD - error)
colors_small = cmap(norm(msds - delta_msds))
plt.scatter(mean_positions[:, 1], mean_positions[:, 0], facecolors='none', edgecolors=colors_small, linestyle='-', linewidths=3, s=point_sizes_small)

plt.title('Mean Square Displacement over 20 frames', fontweight='bold', fontsize=20, y=0.85,x=0.55)
plt.axis('equal')
plt.axis('off')
plt.gca().invert_yaxis()
plt.show()
# %%
#plot 100 random images from dataset in a 10x10 subplot
with open('large_dataset_with_predictions_gaussian_2.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Randomly select 100 rows
image_list = dataset['images']
print(len(image_list))
#%%
# Randomly sample 100 images
random_samples = random.sample(image_list, 100)
#%%
# Create a 10x10 grid of subplots
fig, axes = plt.subplots(10, 7, figsize=(70, 100))

for image, ax in zip(random_samples, axes.ravel()):
    
    # Assuming grayscale images. If they are RGB, you'll need to adjust the code
    ax.imshow(image, cmap='gray',interpolation='none', origin="upper", aspect='equal')
    ax.axis('off')

plt.tight_layout()
plt.show()
# %%
 #Validate performance of segmentation network on the 500 images and plot accuracy as function of noise

# Define the path to your saved checkpoint
file_path = 'best_val_acc_gaussian_2.pth'

# Make sure to load the checkpoint according to your device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
checkpoint = torch.load(file_path, map_location=device)

# Extract the best validation accuracy
best_val_acc = checkpoint.get('best_val_acc', 'Not found')

# Print the best validation accuracy
print(f'Best validation accuracy: {best_val_acc}')
#%%
import numpy as np
import time

with open('benchmark_100_3.pkl', 'rb') as f:
    data = pickle.load(f)
print("i")
image_data = data["benchmark_sets"]
dataframes = data["dataframes"]
lens = []
for dataframe in dataframes:
    lens.append(len(dataframe["label"]))
print(np.mean(lens))
print(dataframes)