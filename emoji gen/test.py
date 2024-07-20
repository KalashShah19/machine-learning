# pip install opencv-python-headless pillow numpy emoji
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np
import emoji

def capture_image(filename='captured_image.jpg'):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Display the resulting frame
        cv2.imshow('Press "s" to save and exit', frame)
        
        # Wait for key press
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Save the image and break the loop
            cv2.imwrite(filename, frame)
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def create_emoji(image_path, output_path='emoji_image.png'):
    # Open the image
    img = Image.open(image_path)
    
    # Resize the image to a smaller size (e.g., 64x64) to simplify it
    img = img.resize((64, 64), Image.ANTIALIAS)
    
    # Apply a filter to make it look more like an emoji (optional)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    
    # Convert to grayscale and then apply a color map (optional)
    img = ImageOps.grayscale(img)
    img = img.convert('RGB')
    
    # Save the resulting emoji-like image
    img.save(output_path)

def add_emoji_overlay(image_path, output_path='emoji_overlay_image.png', emoji_char='😊'):
    # Open the image
    img = Image.open(image_path)
    
    # Create an emoji image
    emoji_img = Image.new('RGBA', img.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(emoji_img)
    d.text((10, 10), emoji.emojize(emoji_char, use_aliases=True), fill=(255, 0, 0, 255))
    
    # Composite the emoji onto the original image
    combined = Image.alpha_composite(img.convert('RGBA'), emoji_img)
    
    # Save the resulting image
    combined.save(output_path)

def main():
    image_path = 'captured_image.jpg'
    emoji_image_path = 'emoji_image.png'
    emoji_overlay_image_path = 'emoji_overlay_image.png'
    emoji_char = '😊'
    
    # Capture image from webcam
    capture_image(image_path)
    
    # Create emoji-like image
    create_emoji(image_path, emoji_image_path)
    
    # Add emoji overlay
    add_emoji_overlay(emoji_image_path, emoji_overlay_image_path, emoji_char)

if __name__ == "__main__":
    main()