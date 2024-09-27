import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
import slideio
from scipy.ndimage import label

def compare_attention_viz(image_name,folder):
    replace_values=["overlap","attention","class"]
    att_arrays=[]
    for val in replace_values:
        folder_temp=folder.replace("overlap",val)
        match=get_matching_images(image_name,folder_temp)
        summed_at,averaged_att=average_attentions(match)
        att_arrays.append(averaged_att)
    visualize_attention_arrays(att_arrays,captions=replace_values)

def average_attentions(attention_list):
    
    summed_att=np.zeros_like(np.load(attention_list[0]))
    count_non_zero= np.zeros_like(summed_att)
    for attention in attention_list:
        att=np.load(attention)
        att=np.nan_to_num(att)
        summed_att=summed_att+att
        count_non_zero=count_non_zero+(att>0).astype(np.uint32)
    return summed_att,summed_att/count_non_zero,count_non_zero

def get_matching_images(image_name,folder):
    return_list=[]
    parts = image_name.split('_')
    key= '_'.join(parts[:-3] + parts[-1:])
    for file in Path(folder).glob("*.npy"):
        parts = str(file).split('_')
        filekey = '_'.join(parts[:-3] + parts[-1:])
        #print(filekey)
        if key in filekey:
            return_list.append(file)
    return return_list

def visualize_attention_array(attention_array, cmap_name='seismic'):
    # Define the colormap (cmap)
    cmap = plt.get_cmap(cmap_name)

    # Set values equal to 0 to NaN to make them white
    attention_array[attention_array == 0] = np.nan

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the attention_array with the chosen colormap
    im = ax.imshow(attention_array, cmap=cmap)

    # Remove axes and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Show the plot
    plt.show()


def visualize_attention_arrays(attention_arrays, captions=None, cmap_name='seismic'):
    # Define the colormap (cmap)
    cmap = plt.get_cmap(cmap_name)

    num_arrays = len(attention_arrays)

    # Set values equal to 0 to NaN to make them white for each attention_array
    for i in range(num_arrays):
        attention_arrays[i][attention_arrays[i] == 0] = np.nan

    # Create a figure and axes for each attention_array
    fig, axes = plt.subplots(1, num_arrays, figsize=(10 * num_arrays, 10))

    for i, ax in enumerate(axes):
        # Plot the attention_array with the chosen colormap
        im = ax.imshow(attention_arrays[i], cmap=cmap)

        # Remove axes and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Add caption if provided
        if captions and i < len(captions):
            ax.text(0.5, -0.1, captions[i], transform=ax.transAxes, ha='center', fontsize=12)

    # Show the plots
    plt.show()

def get_slide(image_name,slide_folder, scaling = 16):
    slide_name,scene_nr=get_slide_name(image_name)
    slide=slideio.Slide(str(Path(slide_folder)/slide_name), "CZI")
    scene = slide.get_scene(scene_nr)

    size=(int(scene.size[0] // scaling), int(scene.size[1] // scaling))
        # read the scene in the desired resolution
    wsi = scene.read_block(
        size=size
    )
    wsi = wsi[..., ::-1]
    orig_size=np.uint32(np.array(size)/8)
    return wsi,orig_size

def get_slide_name(image_name):
    #slide_name=image_name.replace(".png.npy",".czi")
    slide_name_split=image_name.split("_")
    slide_name='_'.join(slide_name_split[:-3]) + ".czi"
    scene_nr=slide_name_split[-1].split(".")[0]
    return slide_name, int(scene_nr)

def overlay(image_name,image_folder,slide_folder,cmap="turbo",save_path=None,multiclass=False,high_att=False):
    #replace_values=["overlap","attention","class"]
    slide,orig_size=get_slide(image_name,slide_folder)
    #for val in replace_values:
    #    folder_temp=image_folder.replace("overlap",val)
    match=get_matching_images(image_name,image_folder)
    save_path_patches=Path(save_path)/"patches"
    save_path_patches.mkdir(parents=True, exist_ok=True)
    summed_at,averaged_att,non_zero=average_attentions(match)
    if multiclass:
        repeated_matrix = np.repeat(np.sum(summed_at,axis=-1)[:, :, np.newaxis], 5, axis=2)

        return highlight_tissue_classes(slide,summed_at/repeated_matrix,image_name,save_path)
    if high_att:
        high_att=get_high_attention_patch((summed_at/np.max(summed_at)),slide_folder,image_name,save_path_patches)
    else:
        high_att =None
    return overlay_colormap_on_rgb(slide,(summed_at/np.max(summed_at)),image_name,save_path,high_att,cmap=cmap)

def overlay_multiclass(image_name,image_folder,attention_folder,slide_folder,save_path=None):
    #replace_values=["overlap","attention","class"]
    slide,orig_size=get_slide(image_name,slide_folder)
    #for val in replace_values:
    #    folder_temp=image_folder.replace("overlap",val)
    match_attention=get_matching_images(image_name,attention_folder)
    match_multiclass=get_matching_images(image_name,image_folder)
    save_path_patches=Path(save_path)/"patches"
    save_path_patches.mkdir(parents=True, exist_ok=True)
    summed_att,averaged_att,_=average_attentions(match_attention)
    summed_multi,averaged_att_multi,_=average_attentions(match_multiclass)
    repeated_matrix = np.repeat(np.sum(summed_multi,axis=-1)[:, :, np.newaxis], 5, axis=2)
    return highlight_tissue_classes(slide,summed_multi/repeated_matrix,image_name,save_path,summed_att/np.max(summed_att))


def get_high_attention_patch(attention_map,slide_folder,image_name,save_path):
    slide,_=get_slide(image_name,slide_folder,2)
    slide=Image.fromarray(slide)
    attention_map=np.fliplr(np.rot90(attention_map, k=3))
    rescaled_attention=Image.fromarray(attention_map).resize((slide.size),Image.NEAREST)

    rescaled_attention=np.array(rescaled_attention)
    max_att=np.where(rescaled_attention>=(np.nanmax(rescaled_attention)*0.99),1,0)

    labeled_array, num_features = label(max_att)

    # Count the number of points in each cluster
    cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Ignoring the background (0)
    largest_cluster_index = np.argmax(cluster_sizes) + 1  # Adding 1 because background is not in bincount

    # Calculate the Center of the Largest Cluster
    largest_cluster = np.where(labeled_array == largest_cluster_index, 1, 0)
    center = np.argwhere(largest_cluster).mean(axis=0)
    center=np.uint64(center)
    x,y=center
    im=np.array(slide)[int(x-512):int(x+512),int(y-512):int(y+512),:]
    Image.fromarray(im).save(Path(save_path)/image_name.replace(".npy",""))
    return center

def highlight_tissue_classes(image, multiclass_values,image_name, save_path,summed_att,alpha=0.5):
    # Create an empty mask to store the highlights
    multiclass_values=np.array(multiclass_values)

    #attention_values[attention_values<0.2]=0

    # Define colors for red, blue, and yellow
    red_color = np.array([255, 0, 0,0])
    blue_color = np.array([0, 0, 255,0])
    yellow_color = np.array([255, 255, 0,0])


    # Calculate weighted colors based on attention values
    weighted_colors = multiclass_values[:, :, 4:5] * red_color + multiclass_values[:, :, 2:3] * blue_color + multiclass_values[:, :, 3:4] * yellow_color

    # Calculate weighted colors based on attention values
    #weighted_colors = attention_values[:, :, 4:5] * red_color +  attention_values[:, :, 3:4] * green_color#+ attention_values[:, :, 2:3] * blue_color #
    weighted_colors[...,-1]=255*alpha
    # Apply the mask to the input image and return the result

    image=Image.fromarray(image)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    im_shape=image.size
    weighted_colors=np.swapaxes(weighted_colors, 0, 1)
   # weighted_colors=np.where(weighted_colors<1,1,weighted_colors)
    summed_att=np.swapaxes(summed_att,0,1)
    colour_mask=Image.fromarray(np.uint8(weighted_colors)).resize(im_shape,Image.BILINEAR)
    mask=Image.fromarray(summed_att).resize(im_shape,Image.BILINEAR)
    zero_array = np.ones_like(colour_mask)*255
    highlighted = Image.alpha_composite(image, colour_mask)
    highlighted = Image.fromarray(np.where(np.expand_dims(np.array(mask),-1), np.array(highlighted), zero_array))
    original = Image.fromarray(np.where(np.expand_dims(np.array(mask),-1), np.array(image), zero_array))
    if save_path is not None:
        highlighted.save(Path(save_path)/image_name.replace(".npy",""))
        original.save(Path(save_path)/image_name.replace(".png.npy","_orig.png"))
    
    return highlighted.convert(image.mode)

def overlay_colormap_on_rgb(rgb_image, heatmap_image, image_name,save_path,high_att_coord, cmap='viridis', alpha=0.4):
    # Apply colormap to heatmap_image
    alpha_channel = np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8) * 255
    
    # Convert RGB image to RGBA format
    rgba_rgb_image = np.dstack((rgb_image, alpha_channel))
    cmap = plt.get_cmap(cmap)
    #inverted_cmap = cmap.reversed()
    heatmap_colored = (cmap(heatmap_image) * 255).astype(np.uint8)
    # Rotate heatmap by 90 degrees
    rotated_heatmap_coloured = np.fliplr(np.rot90(heatmap_colored, k=3))

    # Step 1: Expand the dimensions of resized_binary_mask
    resized_binary_mask = np.rot90(heatmap_image, k=3)


    # Resize rotated_heatmap to match the size of rgb_image
    resized_heatmap = Image.fromarray(rotated_heatmap_coloured).resize((rgb_image.shape[1],rgb_image.shape[0]), Image.BILINEAR)
    resized_binary_mask=Image.fromarray(resized_binary_mask).resize((rgb_image.shape[1],rgb_image.shape[0]), Image.BILINEAR)
    #resized_heatmap.save(image_name.replace(".npy",""))

    # Convert RGB image to PIL format
    pil_rgb_image = Image.fromarray(rgba_rgb_image)

    # Combine the two images with the specified alpha
    overlaid_image = Image.blend(pil_rgb_image, resized_heatmap, alpha)
    resized_binary_mask_expanded=np.expand_dims(np.flip(resized_binary_mask,1),-1)
    #overlaid_image.save("overlay_"+image_name.replace(".npy",""))
    overlaid_image=np.array(overlaid_image)
    zero_array = np.ones_like(overlaid_image)*255
    overlaid_image = np.where(resized_binary_mask_expanded, overlaid_image, zero_array)
    overlaid_image[:,:,3]=255
    #mask = np.all(overlaid_image[:, :, :3] == [153, 153, 153], axis=-1)

    #overlaid_image[mask]=[106, 113, 91,255]
    #overlaid_image=Image.fromarray(overlaid_image)
    overlaid_image=Image.fromarray(overlaid_image,mode='RGBA')
    draw = ImageDraw.Draw(overlaid_image)

    if high_att_coord is not None:
        x,y=high_att_coord
        x,y=int(x/8),int(y/8)
        width = 64  # Replace with your width

        # Calculate coordinates of the right lower corner
        right_lower_corner = (y + width, x + width)

        # Draw a square
        draw.rectangle([y-width, x-width, right_lower_corner], outline="black",width=10)
    if save_path is not None:
        overlaid_image.save(Path(save_path)/image_name.replace(".npy",""))
    return pil_rgb_image


def overlay_colormap_with_alpha(rgb_image, heatmap_image, image_name, save_path, cmap_name='Reds',alpha_v=170):
    # Apply colormap to heatmap_image
    cmap = plt.get_cmap(cmap_name)
    heatmap_colored = (cmap(heatmap_image) * 255).astype(np.uint8)
    heatmap_colored = np.fliplr(np.rot90(heatmap_colored, k=3))
    heatmap_colored[heatmap_colored[..., :3].sum(axis=-1) == 0] = [255, 255, 255, 255] 
    # Calculate alpha values based on whiteness of heatmap pixels
    alpha_values = 1 - np.mean(heatmap_colored, axis=2) / 255.0
    alpha_values = np.clip(alpha_values, 0, 1)
    
    # Convert RGB image to PIL format
    pil_rgb_image = Image.fromarray(rgb_image)

    # Create a copy of the heatmap with adjusted alpha values
    heatmap_with_alpha = heatmap_colored.copy()
    heatmap_with_alpha[..., 3] = (alpha_values * alpha_v).astype(np.uint8)
    
    # Resize the heatmap to match the size of the RGB image
    resized_heatmap = Image.fromarray(heatmap_with_alpha)
    resized_heatmap = resized_heatmap.resize(pil_rgb_image.size, Image.LANCZOS)
    resized_heatmap.save(image_name.replace(".npy", ""))
    # Composite the images
    overlaid_image = Image.alpha_composite(pil_rgb_image.convert("RGBA"), resized_heatmap.convert("RGBA"))
    overlaid_image=np.array(overlaid_image)
    
    overlaid_image=Image.fromarray(overlaid_image,mode='RGBA')
    if save_path is not None:
        overlaid_image.save(Path(save_path) / image_name.replace(".npy", ""))

    return overlaid_image


if __name__ == "__main__":
    done=[]
    image_folder=""
    src_folder=""
    save_folder=""
    #extraction_list=["21-022_K1","21-087_K6","21-083_K1","21-011_K9"]
    for mask in Path(save_folder).glob("*.png"):
        mask_split=str(mask.name).split("_")
        mask_key= "_".join(mask_split[:3]+mask_split[-1:])
        done.append(mask_key)
                
    for mask in Path(src_folder).glob("*.npy"):
        mask_split=str(mask.name).split("_")
        mask_key= "_".join(mask_split[:3]+mask_split[-1:]).replace(".npy","")
        if mask_key not in done:
            try:
                overlay_multiclass(str(mask.name),image_folder,src_folder,"",save_path=save_folder)
            except Exception as e:
                print(e)
            done.append(mask_key)



