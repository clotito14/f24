import qrcode
from PIL import Image

# Step 1: Create a basic QR code
qr = qrcode.QRCode(
    version=1,  # Controls the size of the QR code, try increasing this if image gets too big
    error_correction=qrcode.constants.ERROR_CORRECT_H,  # H = high error correction (allows larger images)
    box_size=10,  # Size of the box where QR code dots are drawn
    border=1,  # Thickness of the border
)

# Step 2: Add data to the QR code
data = "https://siu.presence.io/organization/tau-beta-pi-illinois-epsilon"  # Replace this with your actual data
qr.add_data(data)
qr.make(fit=True)

# Step 3: Create the QR code image
qr_code_img = qr.make_image(fill="black", back_color="white").convert("RGBA")

# Step 4: Open the image you want to place in the center
logo = Image.open("saluki-logo.png")  # Replace with your image path

# Step 5: Calculate the size and position for the logo
qr_code_size = qr_code_img.size
logo_size = (qr_code_size[0] // 3, qr_code_size[1] // 5)  # Resize logo to 1/4th of QR code size
logo = logo.resize(logo_size)

# Step 6: Calculate position to paste the logo in the center of the QR code
logo_position = (
    (qr_code_size[0] - logo_size[0]) // 2,
    (qr_code_size[1] - logo_size[1]) // 2,
)

# Step 7: Paste the logo onto the QR code
qr_code_img.paste(logo, logo_position, mask=logo)

# Step 8: Save or display the final image
qr_code_img.save("saluki-qr.png")
qr_code_img.show()

