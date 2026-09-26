import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def resize(image):
    """Resizes images to a height of 480px to make computation faster"""
    target_height = 480
    height, width = image.shape[:2]
    # Scale width with height if original height!=480px
    if height != target_height:
        scale = target_height / height
        new_width = int(width * scale)

        image = cv2.resize(
            image, (new_width, target_height), interpolation=cv2.INTER_AREA
        )
    return image


def get_keypoints(image):
    """Uses SIFT detector to get keypoints and features"""
    # Convert RGB image to grayscale for feature detection
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize SIFT detector and get keypoints for grayscaled image
    detector = cv2.SIFT_create()
    keypoints, features = detector.detectAndCompute(grayscale_image, None)
    # Draw detected keypoints on original image
    keypoints_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0),
    )
    return keypoints_image, keypoints, features


def get_distance_transform(img_rgb):
    """
    Get distance to closest background pixel for an RGB image
    Input:
        img_rgb: np.array, HxWx3 RGB image
    Output:
        dist: np.array, HxWx1 distance image
            each piexel's intensity is proportional to
            its distance to the closest background pixel
            scaled to [0..255] for plotting
    """
    # Threshold the image: any value above 0 maps into 255
    thresh = cv2.threshold(img_rgb, 0, 255, cv2.THRESH_BINARY)[1]
    # Collapse the color dimension
    thresh = thresh.any(axis=2)
    # Pad to make sure the border is treated as background
    thresh = np.pad(thresh, 1)
    #Get distance transform
    dist = distance_transform_edt(thresh)[1:-1, 1:-1]
    # HxW -> HxWx1
    dist = dist[:, :, None]
    return dist / dist.max() * 255.0


def blend(img1, img2):
    """Blends two warped images using distance transform adaptive weights."""
    # Get weight maps for input images
    weight1 = get_distance_transform(img1)
    weight2 = get_distance_transform(img2)
    # Add total weights for denominator. Make denominator always at least 1.0 to prevent division by 0.
    denom = weight1 + weight2
    denom = np.maximum(denom, 1.0)
    # Get weight average of pixel intensities
    blended = (
        img1.astype(np.float32) * weight1 + img2.astype(np.float32) * weight2
    ) / denom
    # Convert image back to unsigned 8 bit integer
    return blended.astype("uint8")


def get_homography(src, anchor, save_matches_prefix=None):
    """Computes Homography matrix H mapping src_img into anchor_img space."""
    # Get keypoints from src and anchor images
    kp_img_src, kp_src, des_src = get_keypoints(src)
    kp_img_anc, kp_anc, des_anc = get_keypoints(anchor)
    # Use brute force keypoint matching. Errors will be eliminated later by RANSAC.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(des_src, des_anc), key=lambda x: x.distance)

    # Visualize top 50 matches
    drawn_matches = cv2.drawMatches(
        src,
        kp_src,
        anchor,
        kp_anc,
        matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    cv2.imwrite(f"{save_matches_prefix}_matches.jpg", drawn_matches)

    # Format 2D matched point coordinates for RANSAC fitting
    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(
        -1, 2
    )
    pts_anc = np.float32([kp_anc[m.trainIdx].pt for m in matches]).reshape(
        -1, 2
    )

    # Compute Homogrophy using RANSAC with a 1 pixel reprojection error threshold
    H, _ = cv2.findHomography(
        pts_src, pts_anc, method=cv2.RANSAC, ransacReprojThreshold=1.0
    )
    return H


def main():
    paths = [
        "IMG_0820.jpeg",
        "IMG_0821.jpeg",
        "IMG_0822.jpeg",
        "IMG_0823.jpeg",
    ]

    # Load and resize images
    imgs = [resize(cv2.imread(p)) for p in paths]

    # Step 1: Feature Detection & Description
    for idx, img in enumerate(imgs):
        kp_img, keypoints, features= get_keypoints(img)
        cv2.imwrite("Keypoints_Img{idx}.jpg", kp_img)

    # Anchor image setup with imgs[1] as anchor
    anchor_img = imgs[1]
    h_anc, w_anc = anchor_img.shape[:2]
    canvas_w = int(w_anc * 3.5)
    canvas_h = int(h_anc * 1.8)

    offset_x = int(w_anc * 1.2)
    offset_y = int(h_anc * 0.4)
    T = np.array(
        [[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32
    )

    # Pairwise homographies relative to adjacent neighbors
    H0_to_1 = get_homography(
        imgs[0], imgs[1], save_matches_prefix="Pair_0_to_1"
    )
    H2_to_1 = get_homography(
        imgs[2], imgs[1], save_matches_prefix="Pair_2_to_1"
    )
    H3_to_2 = get_homography(
        imgs[3], imgs[2], save_matches_prefix="Pair_3_to_2"
    )

    # Superposition matrix for imgs[3] -> anchor imgs[1]
    H3_to_1 = H2_to_1 @ H3_to_2

    # Warp all individual images to the common anchor image perspective
    warped0 = cv2.warpPerspective(imgs[0], T @ H0_to_1, (canvas_w, canvas_h))
    warped1 = cv2.warpPerspective(imgs[1], T, (canvas_w, canvas_h))
    warped2 = cv2.warpPerspective(imgs[2], T @ H2_to_1, (canvas_w, canvas_h))
    warped3 = cv2.warpPerspective(imgs[3], T @ H3_to_1, (canvas_w, canvas_h))

    cv2.imwrite("Warped_Img0.jpg", warped0)
    cv2.imwrite("Warped_Img1.jpg", warped1)
    cv2.imwrite("Warped_Img2.jpg", warped2)
    cv2.imwrite("Warped_Img3.jpg", warped3)

    # Blend consecutive image pairs
    blend_01 = blend(warped0, warped1)
    blend_12 = blend(warped1, warped2)
    blend_23 = blend(warped2, warped3) 

    cv2.imwrite("Blend_01.jpg", blend_01)
    cv2.imwrite("Blend_12.jpg", blend_12)
    cv2.imwrite("Blend_23.jpg", blend_23)

    # Blend the intermediate pairs
    blend_01_12 = blend(blend_01, blend_12)
    blend_12_23 = blend(blend_12, blend_23)

    cv2.imwrite("Blend_01_12.jpg", blend_01_12)
    cv2.imwrite("Blend_12_23.jpg", blend_12_23)

    # Final pairwise blend to complete final panorama
    final_panorama = blend(blend_01_12, blend_12_23)

    cv2.imwrite("FinalPanorama.jpg", final_panorama)


if __name__ == "__main__":
    main()