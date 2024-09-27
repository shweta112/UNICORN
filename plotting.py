import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import wandb
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont,ImageFile
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import re
from sklearn.metrics import f1_score, accuracy_score
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS=["AIT", "PIT", "EFA", "LFA", "CFA"]


def plot_confusion_matrix(
    gt,
    pred,
    classes,
    save_path,
    normalize=True,
    title="confusion matrix",
    cmap=plt.cm.Blues,
    wandb_run=None
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(list(gt), [item[0][0] for item in pred], labels=np.unique(gt))
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    if normalize: 
        plt.imshow(cm, interpolation="nearest", cmap=cmap,vmin=0,vmax=1)
    else:
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    image_path = Path(save_path) / (title + ".png")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    
    plt.savefig(image_path)
    plt.close()
    # Assuming `image_path` is the path to your image
    image = wandb.Image(str(image_path))
    wandb_run.log({title: image})


def draw_rectangle(draw, rect, color):
    draw.rectangle(rect, fill=color)

def highlight_regions(image, coordinates, attention_values, offset,cmap):
    # Create an empty mask to store the highlights
    mask = Image.new(mode="RGBA", size=image.size, color=(0, 0, 0, 0))

    # Create a draw object to draw the highlight rectangles on the mask
    draw = ImageDraw.Draw(mask)

    # Convert the colormap to 8-bit integers
    cmap_8bit = (cmap(np.array(attention_values))[:, :3] * 255).astype(int)

    # Create a ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor() as executor:
        # Iterate over each coordinate and RGBA color
        for coord, color in zip(coordinates, cmap_8bit):
            # Calculate the coordinates of the region to highlight
            scene, (y, x) = coord
            x2, y2 = min(image.width, x + offset), min(image.height, y + offset)

            # Create a rectangle shape for the region and scale it by the attention value
            rect = ((x, y), (x2, y2))

            # Adjust the alpha value and create the final color
            final_color = tuple(color) + (127,)

            # Submit the drawing task to the executor
            executor.submit(draw_rectangle, draw, rect, final_color)

    # Apply the mask to the input image and return the result
    highlighted = Image.alpha_composite(image.convert("RGBA"), mask)
    return highlighted.convert(image.mode), mask




def compute_attention_overlap(orig_size, coordinates, attention_values,patch_size,class_visualization):
    # Initialize the joined_attentions array with zeros of the same shape as the image
    
    if class_visualization:
        joined_attentions = np.zeros(np.concatenate([orig_size,[5]]))
    else:
        joined_attentions = np.zeros(orig_size)

    patch_size=int(patch_size.cpu().item())
    for coord, attention in zip(coordinates, attention_values):
        # Calculate the coordinates of the region
        scene, y, x = coord.cpu().numpy().astype(np.uint32)
        x2, y2 = x+ patch_size, y + patch_size

        # Create a rectangle shape for the region and scale it by the attention value
        joined_attentions[x:x2,y:y2]=attention

    return joined_attentions


def highlight_tissue_classes(image, coordinates, attention_values, offset):
    # Create an empty mask to store the highlights

    mask = Image.new(mode="RGBA", size=image.size, color=(0, 0, 0, 0))
    if len(attention_values)==0:
        return image,mask
    attention_values=np.array(attention_values)
    #attention_values[attention_values<0.2]=0
    # Create a draw object to draw the highlight rectangles on the mask
    draw = ImageDraw.Draw(mask)

    # Define colors for red, blue, and yellow
    red_color = np.array([255, 0, 0])
    blue_color = np.array([0, 0, 255])
    yellow_color = np.array([255, 255, 0])

    # Calculate weighted colors based on attention values
    weighted_colors = attention_values[:, :, 4] * red_color + attention_values[:, :, 2] * blue_color + attention_values[:, :, 3] * yellow_color
    weighted_colors_8bit = weighted_colors.astype(int)

    with ThreadPoolExecutor() as executor:
        # Iterate over each coordinate and RGBA color
        for coord, color in zip(coordinates, weighted_colors_8bit):
            # Calculate the coordinates of the region to highlight
            scene, (y, x) = coord
            x2, y2 = min(image.width, x + offset), min(image.height, y + offset)

            # Create a rectangle shape for the region and scale it by the attention value
            rect = ((x, y), (x2, y2))

            # Adjust the alpha value and create the final color
            final_color = tuple(color) + (127,)
            # Submit the drawing task to the executor
            executor.submit(draw_rectangle, draw, rect, final_color)

    # Apply the mask to the input image and return the result
    highlighted = Image.alpha_composite(image.convert("RGBA"), mask)
    return highlighted.convert(image.mode), mask

def attention_visualization(attentions, batch, config,save_name="",cmap=plt.cm.viridis,class_visualization=False):
    _, coords, _, patient_name, feature_args, filenames = batch

    for j,(coords_staining,attention) in enumerate(zip(coords,attentions)):
        save_path_mask = Path(config.save_path) /(save_name+"masks")
        save_path_mask.mkdir(parents=True, exist_ok=True)

        save_path_fused = Path(config.save_path) /(save_name+"fused")
        save_path_fused.mkdir(parents=True, exist_ok=True)

        save_path_raw = Path(config.save_path) /(save_name+"raw_attentions")
        save_path_raw.mkdir(parents=True, exist_ok=True)

        coords_rescaled = []
        scene_rescaling = []

        for scene_size in feature_args[j]["original_scene_sizes"][0]:
            downscale_factor = feature_args[j]["downscaling_factor"]
            preview_size = feature_args[j]["preview_size"]
            rescale_factor = np.max(scene_size.cpu().numpy()) / (
                preview_size * downscale_factor
            )
            scene_rescaling.append(rescale_factor)

        for coord in coords_staining[0]:
            scene_index = int(coord[0].item())
            rescaled_yx = coord[1:] / scene_rescaling[scene_index]
            rescaled_coord = (scene_index, rescaled_yx.cpu().numpy())
            coords_rescaled.append(rescaled_coord)

        for scene in range(len(feature_args[j]["original_scene_sizes"][0])):
            offset = feature_args[j]["patch_size"] / scene_rescaling[scene]
            img_name = filenames[j][0].replace(".h5", "_" + str(scene) + ".png")
            preview_name = Path(config.preview_dir) / img_name
            preview = Image.open(preview_name)
            filter_indeces = [i for i, el in enumerate(coords_rescaled) if el[0] == scene]
            filtered_attention=[attention[i] for i in filter_indeces]
            coords_filtered=[coords_rescaled[i] for i in filter_indeces]

            if not class_visualization:
                fused, mask = highlight_regions(
                    preview,
                    coords_filtered,
                    filtered_attention,
                    offset.cpu().numpy(),
                    cmap=cmap
                )
            else:
                print(patient_name)
                fused, mask = highlight_tissue_classes(
                    preview,
                    coords_filtered,
                    filtered_attention,
                    offset.cpu().numpy()
                )
            np.save(save_path_raw/ img_name.replace(".png",""), filtered_attention)
            mask.save(save_path_mask / img_name)
            fused.save(save_path_fused / img_name)

    return save_path_fused


def attention_visualization_offset(attentions, batch, config,save_name="overlap_",class_visualization=False):
    _, coords, _, patient_name, feature_args, filenames = batch

    for j,(coords_staining,attention) in enumerate(zip(coords,attentions)):
        save_path_mask = Path(config.save_path) / config.name/(save_name+"masks")
        save_path_mask.mkdir(parents=True, exist_ok=True)
        downscaling=feature_args[j]['downscaling_factor'].item()
        for scene,orig_size in enumerate((feature_args[j]["original_scene_sizes"][0])):
            img_name = filenames[j][0].replace(".h5", "_" + str(scene) + ".png")

            filter_indeces = [i for i, el in enumerate(coords_staining[0]) if el[0] == scene]

            joined_attentions = compute_attention_overlap(
                (orig_size.cpu().numpy()/downscaling).astype(np.uint32),
                [coords_staining[0][i] for i in filter_indeces],
                [attention[i] for i in filter_indeces],
                feature_args[j]['patch_size'],
                class_visualization=class_visualization
            )
            downscaled=linear_downscale(joined_attentions,8)
            np.save(save_path_mask / img_name,downscaled)
            #Image.fromarray(joined_attentions).save(save_path_mask / img_name)


def linear_downscale(arr, n):
    """
    Linearly downscale a NumPy array by a factor of n along the first two dimensions.
    If the input array is 2D, it will be downscaled as-is.
    If the input array is 3D, the third dimension will be kept unchanged in shape while
    averaging over the first two dimensions.

    Parameters:
        arr (numpy.ndarray): Input array to be downscaled.
        n (int): Downsampling factor.

    Returns:
        numpy.ndarray: Downscaled array.
    """
    if n <= 0:
        raise ValueError("Downsampling factor (n) should be greater than 0.")
    
    if len(arr.shape) == 2:  # Check if input is 2D
        # Calculate new shape after downsampling
        new_shape = (arr.shape[0] // n, arr.shape[1] // n)
        
        # Reshape the input array to enable efficient downsampling
        reshaped_arr = arr[:new_shape[0] * n, :new_shape[1] * n].reshape(new_shape[0], n, new_shape[1], n)
        
        # Take the mean along specified axes to perform downsampling
        downscaled_arr = reshaped_arr.mean(axis=(1, 3))
    elif len(arr.shape) == 3:  # Check if input is 3D
        # Calculate new shape after downsampling
        new_shape = (arr.shape[0] // n, arr.shape[1] // n, arr.shape[2])
        
        # Reshape the input array to enable efficient downsampling
        reshaped_arr = arr[:new_shape[0] * n, :new_shape[1] * n, :].reshape(new_shape[0], n, new_shape[1], n, arr.shape[2])
        
        # Take the mean along the first and second axes to perform downsampling
        downscaled_arr = reshaped_arr.mean(axis=(1, 3))
    else:
        raise ValueError("Input array must be 2D or 3D.")
    
    return downscaled_arr


def class_visualization(attentions, batch,class_predictions, config,save_name="multi_class_",cmap=plt.cm.viridis,):
    _, coords, _, _, _, feature_args, filenames = batch
    attention_offset=0
    for j,coords_staining in enumerate(coords):
        save_path_mask = Path(config.save_dir) / config.name/(save_name+"masks")
        save_path_mask.mkdir(parents=True, exist_ok=True)

        save_path_fused = Path(config.save_dir) / config.name/(save_name+"fused")
        save_path_fused.mkdir(parents=True, exist_ok=True)

        coords_rescaled = []
        scene_rescaling = []

        for scene_size in feature_args[j]["original_scene_sizes"][0]:
            downscale_factor = feature_args[j]["downscaling_factor"]
            preview_size = feature_args[j]["preview_size"]
            rescale_factor = np.max(scene_size.cpu().numpy()) / (
                preview_size * downscale_factor
            )
            scene_rescaling.append(rescale_factor)

        for coord in coords_staining[0]:
            scene_index = int(coord[0].item())
            rescaled_yx = coord[1:] / scene_rescaling[scene_index]
            rescaled_coord = (scene_index, rescaled_yx.cpu().numpy())
            coords_rescaled.append(rescaled_coord)

        for scene in range(len(feature_args[j]["original_scene_sizes"][0])):
            offset = feature_args[j]["patch_size"] / scene_rescaling[scene]
            img_name = filenames[j][0].replace(".h5", "_" + str(scene) + ".png")
            preview_name = Path(config.preview_dir) / img_name
            preview = Image.open(preview_name)
            filter_indeces = [i for i, el in enumerate(coords_rescaled) if el[0] == scene]
            fused, mask = highlight_regions(
                preview,
                [coords_rescaled[i] for i in filter_indeces],
                [attentions[i+attention_offset] for i in filter_indeces],
                offset.cpu().numpy(),
                cmap=cmap
            )

            mask.save(save_path_mask / img_name)
            fused.save(save_path_fused / img_name)
        attention_offset+=len(coords_staining[0])
    return save_path_fused


def plot_distribution(input_list, save_path):

    mapping = {0: 'AIT', 1: 'PIT', 2: 'EFA', 3: 'LFA', 4: 'CFA'}

    mapped_list = [mapping[i] for i in input_list]
    # Counting occurrences of each value
    occurrences = {value: mapped_list.count(value) for value in mapping.values()}

    # Plotting the occurrences
    fig, ax = plt.subplots()
    ax.bar(occurrences.keys(), occurrences.values())

    # Adding the total number of occurrences for each class
    for key, value in occurrences.items():
        ax.text(key, value, str(value), ha='center', va='bottom', fontsize=12)

    # Adding labels and title
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count')
    ax.set_title('Occurrences')

    save_path_distribution=os.path.join(save_path, 'data_distribution.png')

    plt.savefig(save_path_distribution)
    plt.close()
    return save_path_distribution


def create_report(gt,pred,save_path,wandb_run):
    # Plot data distribution
    f1 = f1_score(list(gt), list(pred), average='weighted')
    accuracy = accuracy_score(list(gt), list(pred))
    # Find all .png files with 'confusion' and 'matrix' in their name

    png_files = [f for f in os.listdir(save_path) if re.search(r'confusion(?!.*binary).*matrix.*\.png', f.lower())]
    distribution_plot_path=plot_distribution(gt, save_path)
    png_files.append(distribution_plot_path)
    # Concatenate images
    images = [Image.open(os.path.join(save_path, f)) for f in png_files]
    widths, heights = zip(*(i.size for i in images[:2]))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a white background for the concatenated image
    new_im = Image.new('RGB', (total_width, max_height * 2 + 80), "white")


    # Add images to the new_im
    x_offset = 0
    for im in images[:2]:
        new_im.paste(im, (x_offset, 60))
        x_offset += im.size[0]

    x_offset = 0
    for im in images[2:]:
        new_im.paste(im, (x_offset, max_height + 80))
        x_offset += im.size[0]

 # Draw F1 score and accuracy
    draw = ImageDraw.Draw(new_im)
    font = ImageFont.truetype('/arial.ttf', 30)
    text = f'F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}'
    draw.text((10, max_height + 50), text, font=font, fill=(0, 0, 0))

    # Add title

        
    title = "Model Performance Summary"

    title_font = ImageFont.truetype('/arial.ttf', 44)
    title_size = draw.textsize(title, font=title_font)
    title_position = ((total_width - title_size[0]) // 2, 10)
    draw.text(title_position, title, font=title_font, fill=(0, 0, 0))

    filename='report.png'

    save_path_report=save_path/filename
    new_im.save(save_path_report)
    wandb_run.log({title: wandb.Image(str(save_path_report))})

def create_all_reports(model_outputs,save_path,wandb_run):
    create_report(model_outputs["ground_truth"].values,model_outputs["predictions"].values,save_path,wandb_run)
    
def concatenate_images(images):
    if not images:
        return None
    
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    concatenated_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        concatenated_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return concatenated_image


def create_report_card(staining_contributions,vis_path, patient, probabilities, save_path, ground_truth,font_size = 30, whitespace_height = 200, target_width = 1024,border_left=50):
    save_path.mkdir(exist_ok=True,parents=True)
    patient = patient[0]
    vis_path = Path(vis_path)
    img_paths = [f for f in vis_path.glob("*.png") if patient+"_" in str(f)]

    images_he = []
    images_vk = []
    images_evg = []
    images_movat = []

    for i,img_path in enumerate(img_paths):
        im = Image.open(img_path)
        aspect_ratio = im.width / im.height
        new_height = int(target_width / aspect_ratio)
        im_resized = im.resize((target_width, new_height), Image.LANCZOS)

        if "HE" in img_path.name:
            images_he.append(im_resized)
        elif "vK" in img_path.name:
            images_vk.append(im_resized)
        elif "EvG" in img_path.name:
            images_evg.append(im_resized)
        elif "Movat" in img_path.name:
            images_movat.append(im_resized)

    concatenated_he = concatenate_images(images_he)
    concatenated_vk = concatenate_images(images_vk)
    concatenated_movat = concatenate_images(images_movat)
    concatenated_evg = concatenate_images(images_evg)

    filtered_images=[im for im in [concatenated_he,concatenated_vk,concatenated_movat,concatenated_evg] if im is not None]
    stainings = ["HE", "vK", "Movat","EvG", ]
    used_stainings=[stainings[i] for i,im in enumerate([concatenated_he,concatenated_vk,concatenated_movat,concatenated_evg]) if im is not None]

    # Concatenate images vertically
    widths = [im.width for im in filtered_images]
    heights = [im.height for im in filtered_images]
    max_width = max(widths)
    total_height = sum(heights)
    font=ImageFont.truetype('arial.ttf',font_size)
    concatenated_image = Image.new('RGB', (max_width+border_left, total_height),color='black')
    
    y=0
    draw_concat = ImageDraw.Draw(concatenated_image)
    for used_staining,im in zip(used_stainings,filtered_images):
        concatenated_image.paste(im, (border_left, y))
        draw_concat.text((0,int(y+(im.height/2))), used_staining, fill='white', font=font, anchor=None, spacing=4, align="center")
        draw_concat.text((0,int(y+25+(im.height/2))), str(int(np.round(staining_contributions[used_staining]*100)))+"%", fill='white', font=font, anchor=None, spacing=4, align="center")
        y+=im.height



    # Add whitespace on top of concatenated image
    
    img_with_whitespace = Image.new('RGB', (max_width, total_height + whitespace_height), color='white')
    img_with_whitespace.paste(concatenated_image, (0, whitespace_height))
    # Write prediction and true label on whitespace
    draw = ImageDraw.Draw(img_with_whitespace)

    probabilities=probabilities.cpu().numpy()
    predicted_label = np.argmax(probabilities)
    predicted_probability=np.round(np.max(probabilities)*100)
    
    second_predicted_label=np.argpartition(probabilities.flatten(), -2)[-2]
    second_predicted_probability=probabilities[0][second_predicted_label][0]
    second_predicted_probability=np.round(second_predicted_probability*100)



    correct_prediction = ground_truth.item() == predicted_label

    font_color_multi = 'green' if correct_prediction else 'red'


    prediction_text = f"Predicted: {LABELS[predicted_label]}, {predicted_probability} % ({LABELS[second_predicted_label]}, {second_predicted_probability} %) "
    gt_text=f"True Label: {LABELS[ground_truth.item()]}"

    text_x_position = 10
    text_y_position = 50

    font_large=ImageFont.truetype('arial.ttf',font_size+10)
    text_width, text_height = draw.textsize(patient, font=font_large)
    
    draw.text((max_width-300,65), 'Contributions', fill='black', font=font, anchor=None, spacing=4, align="center")
    draw.text((int((img_with_whitespace.size[0]-text_width)/2), text_x_position), patient, fill='black', font=font_large, anchor=None, spacing=4, align="center")
    draw.text((text_x_position, text_y_position), prediction_text, fill=font_color_multi, font=font, anchor=None, spacing=4, align="left")
    draw.text((text_x_position, text_y_position+25), gt_text, fill=font_color_multi, font=font, anchor=None, spacing=4, align="left")

    # Save the image
    report_card_path = save_path / f"{patient}_report_card.png"
    img_with_whitespace.save(report_card_path)

    return report_card_path



