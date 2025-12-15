# 🔥 TPS-Based In-Plane Thermal Conductivity Prediction Framework

A data-driven framework for **correcting and predicting in-plane thermal conductivity (λ)** of membrane materials  
measured using the **HOTDISK TPS 2500s** system.

The recommended probe radius is **≈ 2 mm**, with a typical **temperature rise of ~10 K** during measurement.  
This framework provides an integrated **machine learning pipeline** that performs:

- 📊 **Data preprocessing and feature engineering**  
- ⚙️ **Systematic error compensation based on measurement parameters**  
- 🧠 **Two-stage calibration model** for accurate prediction of corrected thermal conductivity (`λ_hat`)

## Developer
Dr. Xinhang Jin, Stockholm University, Sweden 
Contact mail:xinhang.jin@su.se

Dr. Xixi Luo, Changan University, China 
Contact mail:xixiluo@chd.edu.cn

## Acknowledgements

Thanks Prof. Aji P. Mathew from Stockholm University as the supervisor of this porject. 

The developer gratefully acknowledge the financial support provided by the FLAG-ERA JTC 2023 program. This work was carried out within the framework of the FLAG-ERA Joint Transnational Call 2023, which has significantly contributed to the advancement of our research.
 
## 🚀 How to use it
First, obtain the time–temperature data from the Hot Disk TPS 2500S instrument.
Tip: During the Hot Disk experiment setup, you may use a nominal (fake) thickness to finish the measurement experiment. For example, if the actual specimen thickness is 50 µm, you can set the thickness to 200 µm in the instrument configuration.

Next, ensure that the following experimental parameters are recorded for subsequent correlation and analysis:

h – The actual thickness of the sample (µm)

rho– The density of the sample (kg/m³)

Cp – The specific heat capacity of the sample (J/kg·K)

or you can directly input Cv - The volume heat capacity of the sample (J/m³.K)

P – The applied heating power (W)

t – The heating duration (s)



To running this program, you can use the following code:
python main.py --data ./data/sample1.xlsx \
  --rho 1200 --Cp 1500 \
  --h 200e-6 --r 2e-3 \
  --P0 0.1 --t_end 5

--data ./data/example.xlsx --Cv 4.0e6 --h 200e-6 --r 2e-3 --P0 0.1 --t_end 5
<img width="1919" height="1023" alt="image" src="https://github.com/user-attachments/assets/14625d66-69b0-4bcf-8a88-179b507b6dc9" />

## 🚀 Key Features
- Two-stage learning pipeline: **system error correction → residual compensation**
- Automatic feature extraction from experimental parameters (e.g., `h`, `r`, `P0`, `dT`, `Q`, `CV`)
- Support for multiple models: **Random Forest**, **SVM**, **1D-CNN**
- Built-in evaluation metrics: **MAE**, **RMSE**, **MAPE**, **sMAPE**
- Modular design for easy integration into lab data pipelines
---

## 📦 Installation
 clone this project from GitHub to your local machine:
```bash
git clone https://github.com/yourname/TPS-ThermalConductivity-ML.git
cd TPS-ThermalConductivity-ML
 

### Requirements
- Python ≥ 3.8  
- Packages: see `requirements.txt`

### Setup
```bash
# Clone the repository
git clone https://github.com/yourname/TPS-ThermalConductivity-ML.git
cd TPS-ThermalConductivity-ML

# Install dependencies

pip install -r requirements.txt

