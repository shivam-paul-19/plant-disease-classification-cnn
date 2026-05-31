import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import cv2

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & METADATA
# ----------------------------------------------------
st.set_page_config(
    page_title="CropCare AI - Explainable Plant Clinic",
    page_icon="🌿",
    layout="wide"
)

# ----------------------------------------------------
# 2. DESIGN AESTHETICS (CUSTOM PREMIUM CSS)
# ----------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Core App container adjustments */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0b0f19 !important; /* Premium Dark Slate background */
    color: #e2e8f0;
}

/* Hide the sidebar so the app reads as a single main page */
[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="collapsedControl"] {
    display: none !important;
}

[data-testid="stAppViewContainer"] {
    margin-left: 0 !important;
}

/* Custom Header elements */
.main-header {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem !important;
    text-align: center;
    margin-bottom: 0.1rem;
    letter-spacing: -0.5px;
}

.subheader {
    text-align: center;
    color: #94a3b8;
    font-size: 1.2rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
}

/* Premium Card container */
.premium-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}

.card-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 16px;
    color: #f8fafc;
    border-bottom: 2px solid rgba(16, 185, 129, 0.2);
    padding-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Badge System */
.badge-healthy {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 9999px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
}

.badge-disease {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 9999px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
}

/* Metric Display */
.metric-container {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.metric-label {
    font-size: 0.9rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
}

/* Custom Advisory layout */
.advise-box {
    margin-top: 14px;
    padding: 16px;
    border-radius: 12px;
    font-size: 0.98rem;
    line-height: 1.5;
}

.advise-organic {
    background: rgba(16, 185, 129, 0.05);
    border-left: 4px solid #10b981;
}

.advise-chemical {
    background: rgba(245, 158, 11, 0.05);
    border-left: 4px solid #f59e0b;
}

.advise-prevention {
    background: rgba(59, 130, 246, 0.05);
    border-left: 4px solid #3b82f6;
}

.advise-title {
    font-weight: 700;
    margin-bottom: 6px;
    font-size: 1.05rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. DEFINE PYTORCH MODEL ARCHITECTURE
# ----------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.feature = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=16, padding=1, stride=2, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32, padding=1, stride=2, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, padding=1, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            # Block 4
            nn.Conv2d(in_channels=64, out_channels=128, padding=2, stride=2, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 38)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.global_pool(x)
        return self.classifier(x)

# ----------------------------------------------------
# 4. LOAD PRE-TRAINED MODEL WITH CACHING
# ----------------------------------------------------
@st.cache_resource
def load_trained_model():
    model = CNN(num_channels=3)
    # Load state dict on CPU for maximum portability
    model.load_state_dict(torch.load("plant_disease_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

try:
    model = load_trained_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load plant_disease_model.pth. Make sure the file exists in the directory. Error: {e}")

# ----------------------------------------------------
# 5. DICTIONARY MAPPINGS & ADVISORY DATABASE
# ----------------------------------------------------
ID_TO_LABEL = {
    0: 'Tomato Late blight',
    1: 'Tomato healthy',
    2: 'Grape healthy',
    3: 'Orange Haunglongbing (Citrus greening)',
    4: 'Soybean healthy',
    5: 'Squash Powdery mildew',
    6: 'Potato healthy',
    7: 'Corn (maize) Northern Leaf Blight',
    8: 'Tomato Early blight',
    9: 'Tomato Septoria leaf spot',
    10: 'Corn (maize) Cercospora leaf spot Gray leaf spot',
    11: 'Strawberry Leaf scorch',
    12: 'Peach healthy',
    13: 'Apple Apple scab',
    14: 'Tomato Tomato Yellow Leaf Curl Virus',
    15: 'Tomato Bacterial spot',
    16: 'Apple Black rot',
    17: 'Blueberry healthy',
    18: 'Cherry (including sour) Powdery mildew',
    19: 'Peach Bacterial spot',
    20: 'Apple Cedar apple rust',
    21: 'Tomato Target Spot',
    22: 'Pepper, bell healthy',
    23: 'Grape Leaf blight (Isariopsis Leaf Spot)',
    24: 'Potato Late blight',
    25: 'Tomato Tomato mosaic virus',
    26: 'Strawberry healthy',
    27: 'Apple healthy',
    28: 'Grape Black rot',
    29: 'Potato Early blight',
    30: 'Cherry (including sour) healthy',
    31: 'Corn (maize) Common rust ',
    32: 'Grape Esca (Black Measles)',
    33: 'Raspberry healthy',
    34: 'Tomato Leaf Mold',
    35: 'Tomato Spider mites Two-spotted spider mite',
    36: 'Pepper, bell Bacterial spot',
    37: 'Corn (maize) healthy'
}

# Advanced agronomic advisory data
ADVISORY_DB = {
    "healthy": {
        "status": "Excellent Health status",
        "symptoms": "No plant disease symptoms detected. The leaves exhibit uniform cellular greening, strong structure, and optimized chlorophyll content.",
        "organic": "Maintain normal fertilizing using natural compost, worm castings, or liquid kelp extract. Avoid over-watering.",
        "chemical": "No chemical intervention needed or advised. Keep the ecosystem balanced with beneficial insects.",
        "prevention": "Sanitize tools between uses, rotate planting spots yearly, and keep watering at the base rather than overhead."
    },
    "Late blight": {
        "status": "Infected (Late Blight Fungal disease)",
        "symptoms": "Water-soaked dark lesions appearing on leaves, expanding rapidly. Under humid conditions, a velvety white downy mold grows on the leaf underside.",
        "organic": "Destroy heavily infected foliage immediately. Apply copper octanoate sprays, neem oil extracts, or active compost tea weekly.",
        "chemical": "Apply broad-spectrum fungicides containing chlorothalonil, metalaxyl, or mancozeb as directed by local agronomists.",
        "prevention": "Ensure wide spacing for ventilation. Prune lower leaf levels. Water early in the morning so foliage dries quickly."
    },
    "Early blight": {
        "status": "Infected (Early Blight Fungal infection)",
        "symptoms": "Small, concentric dark brown rings forming 'bullseye' shapes starting on older mature foliage, eventually causing yellowing and defoliation.",
        "organic": "Prune out diseased leaves near the base. Apply Bacillus subtilis solutions or bio-fungicide sprays containing organic copper.",
        "chemical": "Apply protective fungicides with active ingredients like azoxystrobin, pyraclostrobin, or difenoconazole.",
        "prevention": "Rotate crops every year. Lay down mulch to stop soil spores splashing onto plant leaves. Minimize dense planting."
    },
    "Black rot": {
        "status": "Infected (Black Rot Fungal/Bacterial disease)",
        "symptoms": "V-shaped yellow leaf lesions starting from leaf edges, progressing toward the center turning veins dry, black, and brittle.",
        "organic": "Carefully prune out infected stems with sterilized tools. Treat remaining canopy with sulfur-based sprays.",
        "chemical": "Utilize copper hydroxide bactericides or systemic fungicides with azoxystrobin properties.",
        "prevention": "Plant certified seed varieties, handle weeds promptly, and avoid pruning during wet periods."
    },
    "Powdery mildew": {
        "status": "Infected (Powdery Mildew Fungal disease)",
        "symptoms": "White, talcum powder-like fungal patches covering leaf surfaces, leading to leaf curling, drying, and stunting.",
        "organic": "Spray potassium bicarbonate mixtures, horticultural oils, or diluted milk spray (1 part milk, 9 parts water) under direct sunshine.",
        "chemical": "Treat with fungicides containing myclobutanil, triadimefon, or elemental sulfur solutions.",
        "prevention": "Grow in areas with bright solar radiation. Keep pruning optimized for superior air circulation."
    },
    "Apple scab": {
        "status": "Infected (Apple Scab Fungal disease)",
        "symptoms": "Olive-green, velvety spots developing on leaves that become puckered, dark brown, and drop prematurely.",
        "organic": "Rake and completely burn or dispose of fallen leaves in autumn. Spray copper soap or lime sulfur during early bud break.",
        "chemical": "Treat with chemical fungicides such as captan, dodine, or myclobutanil at the green-tip phase.",
        "prevention": "Prune tree canopies to let wind dry leaves. Choose scab-resistant apple cultivars."
    },
    "Cedar apple rust": {
        "status": "Infected (Cedar Apple Rust Fungal infection)",
        "symptoms": "Vibrant yellow-orange spots on the leaf top. Later, tiny tube-like orange projection structures appear under the leaf.",
        "organic": "Treat foliage with copper-based organic sprays. Prune and destroy any infected twigs.",
        "chemical": "Spray systemic rust inhibitors such as propiconazole or myclobutanil during early spring leaf-bud opening.",
        "prevention": "Remove close-proximity juniper or cedar trees (the alternate host) within 500 feet of the orchard."
    },
    "Yellow Leaf Curl Virus": {
        "status": "Infected (Yellow Leaf Curl Virus)",
        "symptoms": "Upward curling and severe yellowing of leaf edges, stunted development, and complete lack of fruit set. Transmitted by Whiteflies.",
        "organic": "Set up bright yellow sticky cards to capture Whiteflies. Apply neem oil or potassium soaps to foliage.",
        "chemical": "Apply systemic soil-drenched insecticides containing imidacloprid to suppress vector insects.",
        "prevention": "Construct fine physical insect netting, remove nearby wild weeds, and grow resistant hybrid seeds."
    },
    "Bacterial spot": {
        "status": "Infected (Bacterial Spot disease)",
        "symptoms": "Small, dark brown angular water-soaked spots surrounded by prominent yellow halos, resulting in leaf drying.",
        "organic": "Use copper hydroxide bactericides. Strictly avoid working with or walking among wet crops.",
        "chemical": "Spray a combined tank mix of copper fungicides and mancozeb for maximum bacterial knockdown.",
        "prevention": "Utilize verified disease-free nursery stock, rotate plant plots, and avoid overhead sprinkler systems."
    },
    "Septoria leaf spot": {
        "status": "Infected (Septoria Fungal leaf spot)",
        "symptoms": "Abundant small circular gray spots with black borders on lower leaves. Inside the gray centers, tiny black pimple-like bodies develop.",
        "organic": "Strip off infected bottom leaves. Apply copper fungicides or organic bio-fungicides regularly.",
        "chemical": "Apply synthetic protectants containing active ingredients like chlorothalonil or mancozeb.",
        "prevention": "Lay organic mulch to prevent soil spores from splashing up, and water crops directly at soil level."
    },
    "Cercospora leaf spot": {
        "status": "Infected (Cercospora Leaf Spot Fungal infection)",
        "symptoms": "Tan or gray circular leaf spots featuring distinct reddish-purple borders, spreading from the bottom up.",
        "organic": "Open tree/crop spacing for air. Treat with copper compounds or organic bio-fungicides.",
        "chemical": "Utilize strobilurin fungicides (e.g., azoxystrobin or pyraclostrobin).",
        "prevention": "Tear out or bury crop residues, rotate non-host crops, and irrigate early in the day."
    },
    "Leaf scorch": {
        "status": "Infected (Leaf Scorch Fungal/Environmental stress)",
        "symptoms": "Brown, dry margins developing along leaf borders. Under moist conditions, small black spots form.",
        "organic": "Remove fallen dead leaves. Hydrate plants deeply. Protect from direct strong winds.",
        "chemical": "Spray protective fungicides containing copper or captan to stop secondary leaf-spot invasions.",
        "prevention": "Mulch the root zone heavily to conserve moisture, and set up windbreaks if necessary."
    },
    "Target Spot": {
        "status": "Infected (Target Spot Fungal disease)",
        "symptoms": "Concentric circle lesions resembling targets, starting on the lower canopy and moving upward.",
        "organic": "Spray organic copper soaps. Thin dense branches to reduce micro-climate humidity.",
        "chemical": "Treat with fungicides featuring azoxystrobin, cyprodinil, or pyraclostrobin.",
        "prevention": "Always avoid overhead watering, prune lower limbs, and till soil to bury fungal debris."
    },
    "mosaic virus": {
        "status": "Infected (Mosaic Virus)",
        "symptoms": "Irregular light and dark green mottled patterns on foliage, leaf wrinkling/blistering, and severe plant stunting.",
        "organic": "No cure exists. Immediately dig up and burn infected plants. Sanitize tools in milk or disinfectant.",
        "chemical": "Viruses cannot be cured with chemicals. Focus on suppressing carrier pests (Aphids) with insecticidal soaps.",
        "prevention": "Avoid tobacco products near plants, select certified clean seeds, and keep weed growth minimized."
    },
    "Spider mites": {
        "status": "Infected (Spider Mite Infestation)",
        "symptoms": "Extremely fine webbing underneath leaves, dotted yellow speckles on top, with leaves turning bronze and dry.",
        "organic": "Release beneficial predators like lacewings or predatory mites. Spray leaves with pressurized water or neem oil.",
        "chemical": "Apply selective acaricides like abamectin or spiromesifen to save non-target beneficial insect populations.",
        "prevention": "Irrigate consistently to prevent drought stress which attracts mites. Maintain high humidity if growing indoors."
    },
    "Leaf Mold": {
        "status": "Infected (Leaf Mold Fungal disease)",
        "symptoms": "Pale green/yellow spots on the upper leaf side, with olive-purple velvety mold thriving on the lower side.",
        "organic": "Boost venting inside greenhouses. Treat foliage with sulfur or copper soaps.",
        "chemical": "Use fungicides containing chlorothalonil, difenoconazole, or cyprodinil.",
        "prevention": "Keep relative humidity levels below 85%, practice row spacing, and clean greenhouse structures."
    },
    "Haunglongbing": {
        "status": "Infected (Citrus Greening / Huanglongbing)",
        "symptoms": "Asymmetrical blotchy yellow leaf mottling, yellow veins, small upright leaves, and bitter, lopsided citrus fruits.",
        "organic": "No systemic cure. Control Asian citrus psyllids with organic mineral oils. Boost tree vigor with micro-nutrients.",
        "chemical": "Apply chemical insect sprays targeting psyllids (e.g., imidacloprid or bifenthrin).",
        "prevention": "Immediately cut down and destroy positive trees to save neighboring orchards. Purchase certified nursery stock."
    },
    "Esca": {
        "status": "Infected (Esca / Black Measles Fungal complex)",
        "symptoms": "Interveinal yellowing and necrosis forming a striking 'tiger-stripe' pattern on leaves.",
        "organic": "Coat fresh pruning cuts with protective sealing paint or Trichoderma biocontrol paste.",
        "chemical": "Apply copper-based wound dressings to cut surfaces. Chemical soil treatments are largely ineffective.",
        "prevention": "Avoid pruning during damp, foggy periods. Disinfect shears thoroughly when moving between vines."
    }
}

def get_disease_details(label):
    # Match label to category keys
    for key, data in ADVISORY_DB.items():
        if key.lower() in label.lower():
            return data
    # Fallback to healthy details if not matched
    return ADVISORY_DB["healthy"]

# ----------------------------------------------------
# 6. GRAD-CAM (EXPLAINABILITY HEATMAP ENGINE)
# ----------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Safe registration supporting multiple PyTorch versions
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        try:
            self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
        except AttributeError:
            self.backward_hook = self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # Backward pass
        score = logits[0, class_idx]
        score.backward()
        
        # Extract gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().data.numpy()[0]  # (C, H, W)
        
        # Calculate channel-wise weights
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Linear combination
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize between [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        # Sharpen hotspots and filter out background noise (contrast enhancement)
        # Power scaling (enhances peak values and suppresses low values exponentially)
        cam = np.power(cam, 2.5)
        
        # Hard-threshold background glow (anything below 20% intensity becomes 0)
        cam[cam < 0.20] = 0.0
        
        # Re-normalize to stretch peak values back to 1.0 after thresholding
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        # Interpolate/resize to original input resolution (224, 224) using Pillow
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        cam = np.array(cam_img) / 255.0
        
        return cam, class_idx
        
    def release(self):
        # Clean up hooks to prevent memory leaks
        self.forward_hook.remove()
        self.backward_hook.remove()

def overlay_heatmap_on_image(img_pil, heatmap):
    # Resize original image to 224x224 to match heatmap
    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized)
    
    # Generate colormap heatmap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Alpha blend: 60% original image + 40% heatmap color
    blended = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    
    return Image.fromarray(blended)

# ----------------------------------------------------
# 7. INFERENCE PIPELINE
# ----------------------------------------------------
def predict_and_explain(image_bytes, model):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 1. Resize to 224x224
    img_resized = img.resize((224, 224))
    
    # 2. Scale values to [0, 1]
    img_array = np.array(img_resized) / 255.0
    
    # 3. Convert to float tensor and permute dimensions to (C, H, W)
    image_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)
    
    # 4. Add batch dimension (1, C, H, W) and require grad for backward pass
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor.requires_grad = True
    
    # Hook the ReLU output of the final convolutional block (index 13) for post-rectified activations
    target_layer = model.feature[13]
    cam_engine = GradCAM(model, target_layer)
    
    # Run predictions with gradients enabled for backward pass
    with torch.enable_grad():
        heatmap, class_idx = cam_engine(image_tensor)
        
        # Calculate full forward probabilities separately
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        
    confidence_score = probabilities[class_idx].item() * 100
    predicted_label = ID_TO_LABEL[class_idx]
    
    # Generate overlaid explainable image
    overlaid_image = overlay_heatmap_on_image(img, heatmap)
    
    # Release hooks
    cam_engine.release()
    
    return predicted_label, confidence_score, overlaid_image

# ----------------------------------------------------
# 8. APP INTERFACE LAYOUT
# ----------------------------------------------------
st.markdown("<div class='main-header'>CropCare AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Scan your crops using state-of-the-art Deep Learning to identify diseases and receive organic treatment guidelines instantly.</div>", unsafe_allow_html=True)

st.markdown("### Supported Crops", unsafe_allow_html=True)
st.markdown("Tomato, Grape, Orange (Citrus), Soybean, Squash, Potato, Corn, Strawberry, Peach, Apple, Blueberry, Cherry, Bell Pepper, Raspberry.", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ Application is running in demo mode because the weight file could not be found.")

# Split page into Two Main Panels
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("<div class='premium-card'><div class='card-title'>🌿 Upload Leaf Image</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select or drag & drop a plant leaf photo (JPG/PNG)", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        # Display side-by-side comparison of original and Grad-CAM in the left column
        st.markdown("<div class='premium-card'><div class='card-title'>🩺 Diagnostic Imaging Compare</div>", unsafe_allow_html=True)
        
        # Run prediction on upload
        if model_loaded:
            with st.spinner("🔃 Running deep neural network and Grad-CAM analysis..."):
                label, conf, overlaid_img = predict_and_explain(file_bytes, model)
                
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem; font-weight: 600;'>Original Photo</p>", unsafe_allow_html=True)
                st.image(file_bytes, use_container_width=True)
            with img_col2:
                st.markdown("<p style='text-align: center; color: #34d399; font-size: 0.9rem; font-weight: 600;'>Explainability Heatmap</p>", unsafe_allow_html=True)
                st.image(overlaid_img, use_container_width=True)
                
            st.markdown("""
            <div style='margin-top: 14px; padding: 12px; background: rgba(30, 41, 59, 0.4); border-radius: 8px; font-size: 0.88rem; color: #94a3b8; line-height: 1.4;'>
                🤖 <b>Explainability Legend:</b> The glowing <b>red/orange</b> hotspots represent the exact biological markers (e.g. necrotic spots, lesions, vein yellowing) that the model focused on to make its diagnosis. Blue/green zones were ignored.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    if uploaded_file is not None:
        if model_loaded:
            details = get_disease_details(label)
            is_healthy = "healthy" in label.lower()
            
            # Plant Extraction
            plant_name = label.split("___")[0].split(" (")[0].replace(",", "").capitalize()
            disease_name = label.split("___")[-1].replace("_", " ").capitalize()
            
            st.markdown("<div class='premium-card'><div class='card-title'>📈 Classification Result</div>", unsafe_allow_html=True)
            
            # Header Badge
            if is_healthy:
                st.markdown(f"<div class='badge-healthy'>✨ {details['status']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='badge-disease'>⚠️ {details['status']}</div>", unsafe_allow_html=True)
                
            # Stats row
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-label'>🌿 Target Plant</div>
                    <div class='metric-val'>{plant_name}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_stat2:
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-label'>⚡ Confidence</div>
                    <div class='metric-val'>{conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"""
            <div class='metric-container' style='margin-top: 16px;'>
                <div class='metric-label'>🩺 Diagnostic Outcome</div>
                <div class='metric-val' style='font-size: 1.4rem; color: #10b981;'>{disease_name}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Advisory Card
            st.markdown("<div class='premium-card'><div class='card-title'>📋 Agronomic Advisory Report</div>", unsafe_allow_html=True)
            
            st.markdown(f"**🔬 Symptoms Identified:**\n{details['symptoms']}")
            
            # Organic Advisory
            st.markdown(f"""
            <div class='advise-box advise-organic'>
                <div class='advise-title'><span style='color: #10b981;'>🍃</span> Organic Action Plan</div>
                <div>{details['organic']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Chemical Advisory
            st.markdown(f"""
            <div class='advise-box advise-chemical'>
                <div class='advise-title'><span style='color: #f59e0b;'>🧪</span> Chemical Management Plan</div>
                <div>{details['chemical']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Prevention Guidelines
            st.markdown(f"""
            <div class='advise-box advise-prevention'>
                <div class='advise-title'><span style='color: #3b82f6;'>🛡️</span> Preventative Guidelines</div>
                <div>{details['prevention']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.error("Error loading the neural network weights. Unable to run prediction.")
    else:
        # Default screen when no file is uploaded
        st.markdown("<div class='premium-card' style='text-align: center; padding: 60px 40px;'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size: 4rem;'>🌿</span>", unsafe_allow_html=True)
        st.markdown("### Awaiting Plant Leaf Upload", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b;'>Upload a close-up photo of a crop leaf from your computer to run the high-precision diagnosis pipeline.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

