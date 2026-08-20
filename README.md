# TPS In-Plane Thermal Conductivity Fitting Software

## 1. Overview

This software is designed for fitting the **in-plane thermal conductivity of thin films and membrane materials** using transient plane source (TPS) time–temperature data.

The program consists of two main components:

```text
TPS_Project/
│
├── main.py
│
└── TPS_Fitting_CPU_Kernel_ABC.py
```

* `main.py` — graphical user interface (GUI)
* `TPS_Fitting_CPU_Kernel_ABC.py` — TPS calculation and fitting kernel

The software performs TPS curve fitting to obtain the initial fitted thermal conductivity, `λ_hat`, and subsequently applies a thickness correction to obtain the final corrected in-plane thermal conductivity, `λ_corrected`.

---

## 2. Main Features

The software provides:

* Graphical user interface for TPS data analysis
* Import of experimental time–temperature data
* CPU-based TPS numerical calculation
* Automatic thermal conductivity fitting
* Three-stage thermal conductivity search
* User-defined sample and experimental parameters
* Thickness-dependent correction
* Adjustable correction parameters `A`, `B`, and `C`
* Comparison between experimental and calculated temperature-rise curves
* Output of fitted and corrected thermal conductivity
* Output of relative fitting error
* Export of fitting parameters
* Export of time–temperature and fitted curve data

---

## 3. Thickness Correction

The TPS kernel first determines the fitted thermal conductivity:

[
\lambda_{\mathrm{hat}}
]

A thickness-dependent correction is then applied according to:

[
E_h=\frac{A}{(h+B)^2}+C
]

where:

* (E_h) is the thickness-related error in W m⁻¹ K⁻¹
* (h) is the sample thickness in μm
* (A), (B), and (C) are correction parameters

The corrected thermal conductivity is calculated as:

[
\lambda_{\mathrm{corrected}}
============================

\lambda_{\mathrm{hat}}-E_h
]

or:

[
\boxed{
\lambda_{\mathrm{corrected}}
============================

## \lambda_{\mathrm{hat}}

\left[
\frac{A}{(h+B)^2}+C
\right]
}
]

### Default correction parameters

```text
A = 160372
B = 218 μm
C = -0.03 W m⁻¹ K⁻¹
```

These values are used automatically when the software starts.

The user can modify `A`, `B`, and `C` directly through the graphical interface.

---

## 4. Required Python Environment

Python 3.10 or later is recommended.

Required packages:

```text
numpy
pandas
scipy
matplotlib
openpyxl
```

`Tkinter` is used for the graphical interface and is normally included with standard Python installations.

Install the required third-party packages using:

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

## 5. Running the Software

Place the following files in the same directory:

```text
main.py
TPS_Fitting_CPU_Kernel_ABC.py
```

Open a terminal in this directory and run:

```bash
python main.py
```

The graphical interface will then open.

---

## 6. Experimental Data

The software is intended to process TPS experimental **time–temperature data**.

The basic data structure is:

|  Time | Temperature |
| ----: | ----------: |
| (t_1) |       (T_1) |
| (t_2) |       (T_2) |
| (t_3) |       (T_3) |
|   ... |         ... |

The first column represents time and the second column represents measured temperature.

The GUI supports importing experimental data from compatible Excel or CSV files.

Before fitting, verify that:

* time values are numerical;
* temperature values are numerical;
* the data are arranged in chronological order;
* the experimental parameters correspond to the imported measurement;
* sufficient valid data points are available.

---

## 7. Input Parameters

The main experimental parameters include:

### Heating power — `P0`

Heating power applied by the TPS probe.

The kernel accepts the input according to its predefined unit conversion.

### Volumetric heat capacity — `Cv`

Volumetric heat capacity of the sample.

### Probe radius — `r`

Radius of the TPS sensor/probe.

### Sample thickness — `h`

Thickness of the membrane or thin-film sample.

This parameter is particularly important because it is used both in the TPS model and in the thickness correction.

### Analysis time — `t_end`

Defines the maximum experimental time included in the fitting.

### Correction parameters — `A`, `B`, `C`

These parameters control the thickness correction:

[
E_h=\frac{A}{(h+B)^2}+C
]

Default values:

```text
A = 160372
B = 218
C = -0.03
```

Users may replace these values when another calibrated correction model is required.

---

## 8. TPS Fitting Procedure

The program automatically searches for the thermal conductivity that provides the best agreement between the calculated TPS response and the experimental temperature-rise curve.

The fitting kernel uses a three-stage search.

### Stage 1 — Coarse search

```text
Step = 1 W m⁻¹ K⁻¹
```

This stage identifies the approximate thermal conductivity range.

### Stage 2 — Medium search

```text
Step = 0.1 W m⁻¹ K⁻¹
```

A narrower range around the result from Stage 1 is evaluated.

### Stage 3 — Fine search

```text
Step = 0.01 W m⁻¹ K⁻¹
```

The final thermal conductivity is determined with higher numerical resolution.

The resulting value before thickness correction is reported as:

```text
lambda_hat_W_mK
```

---

## 9. Fitting Error

The software calculates the difference between the experimental temperature response and the TPS model response.

A relative fitting error is reported as:

```text
relative_fit_error
```

A smaller value generally indicates better agreement between the experimental and calculated temperature-rise curves.

The fitting error should always be considered together with visual inspection of the fitted curve.

---

## 10. Output Results

After successful calculation, the software provides the main results including:

```text
lambda_hat_W_mK
lambda_corrected_W_mK
relative_fit_error
A
B_um
C_W_mK
```

### `lambda_hat_W_mK`

Thermal conductivity directly obtained from TPS curve fitting before thickness correction.

### `lambda_corrected_W_mK`

Final thermal conductivity after applying the thickness correction.

This is calculated as:

[
\lambda_{\mathrm{corrected}}
============================

## \lambda_{\mathrm{hat}}

\left[
\frac{A}{(h+B)^2}+C
\right]
]

### `relative_fit_error`

Relative error between the calculated and experimental temperature-rise responses.

---

## 11. Time–Temperature Data Interface

The TPS kernel also provides an interface for retrieving the experimental and calculated curve data.

The returned data include:

```text
time_s
temperature_K
experimental_delta_T_K
calculated_delta_T_K
```

where:

* `time_s` — experimental time
* `temperature_K` — measured temperature
* `experimental_delta_T_K` — experimental temperature rise used for fitting
* `calculated_delta_T_K` — temperature rise predicted by the TPS model

These data can be used for:

* plotting;
* checking fitting quality;
* statistical analysis;
* external data processing;
* publication figures;
* comparison between different samples.

---

## 12. Graphical Curve Display

The graphical interface plots the experimental and calculated TPS responses for direct comparison.

The principal curves are:

[
\Delta T_{\mathrm{experimental}}(t)
]

and

[
\Delta T_{\mathrm{calculated}}(t)
]

A good fitting result should show close agreement between the two curves within the selected fitting interval.

Visual inspection is recommended even when the numerical fitting error is small.

---

## 13. Exporting Results

The software provides interfaces for exporting two types of results.

### Parameter results

The parameter output contains information such as:

```text
lambda_hat
lambda_corrected
A
B
C
relative_fit_error
```

This file can be used for statistical analysis or comparison among different samples.

### Time–temperature results

The curve-data output contains:

```text
time_s
temperature_K
experimental_delta_T_K
calculated_delta_T_K
```

This allows the fitting curves to be reproduced independently using Origin, MATLAB, Python, Excel, or other plotting software.

---

## 14. Recommended Workflow

A typical analysis procedure is:

```text
Experimental TPS measurement
        ↓
Import time–temperature data
        ↓
Enter experimental parameters
        ↓
Check P0, Cv, r, h and t_end
        ↓
Set A, B and C
        ↓
Run TPS fitting
        ↓
Obtain λ_hat
        ↓
Calculate thickness correction
        ↓
Obtain λ_corrected
        ↓
Inspect experimental/fitted curves
        ↓
Check relative fitting error
        ↓
Export parameters and curve data
```

---

## 15. Important Notes

### Units

Special attention should be paid to units when entering experimental parameters.

Incorrect units for heating power, volumetric heat capacity, probe radius, or sample thickness can produce physically incorrect thermal conductivity results.

### Thickness

The thickness value must correspond to the actual thickness of the sample used during the TPS experiment.

Because thickness is explicitly included in the correction model, an incorrect thickness affects both the TPS calculation and the final corrected thermal conductivity.

### Correction parameters

The default parameters are:

```text
A = 160372
B = 218 μm
C = -0.03 W m⁻¹ K⁻¹
```

These parameters may be changed by the user.

When comparing multiple samples within the same calibrated measurement framework, the same correction model should normally be used unless there is a justified reason to change it.

### Fitting quality

Do not evaluate fitting quality using only the final thermal conductivity value.

Always check:

1. experimental and calculated curves;
2. relative fitting error;
3. experimental parameter units;
4. selected analysis time;
5. sample thickness;
6. physical plausibility of the fitted conductivity.

---

## 16. Project Structure

Recommended project structure:

```text
TPS_Project/
│
├── main.py
├── TPS_Fitting_CPU_Kernel_ABC.py
│
├── Data/
│   ├── sample_01.xlsx
│   ├── sample_02.xlsx
│   └── ...
│
└── Results/
    ├── fitting_parameters.csv
    └── time_temperature.csv
```

The GUI and TPS kernel should remain in the same directory unless the Python import path in `main.py` is modified accordingly.

---

## 17. Troubleshooting

### The GUI does not start

Check that Python is installed:

```bash
python --version
```

Then verify the required packages:

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

### `ModuleNotFoundError: TPS_Fitting_CPU_Kernel_ABC`

Make sure:

```text
main.py
TPS_Fitting_CPU_Kernel_ABC.py
```

are located in the same directory.

### Excel data cannot be loaded

Verify that `openpyxl` is installed:

```bash
pip install openpyxl
```

Also check that the selected file contains valid numerical time and temperature data.

### Fitting result appears unreasonable

Check the following first:

* heating power `P0`;
* volumetric heat capacity `Cv`;
* probe radius `r`;
* sample thickness `h`;
* fitting time range;
* imported time–temperature columns;
* units of all parameters;
* correction parameters `A`, `B`, and `C`.

### Corrected thermal conductivity differs strongly from `λ_hat`

Check the magnitude of:

[
\frac{A}{(h+B)^2}+C
]

The correction becomes more significant for thin samples.

---

## 18. Core Python Interface

The TPS calculation can also be called directly without the graphical interface.

Example:

```python
from TPS_Fitting_CPU_Kernel_ABC import fit_tps_kernel

result = fit_tps_kernel(
    time_s,
    temperature_k,
    P0=20,
    Cv=2.1,
    r=2.0,
    h=100,
    A=160372,
    B=218,
    C=-0.03,
)

print(result["lambda_hat_W_mK"])
print(result["lambda_corrected_W_mK"])
print(result["relative_fit_error"])
```

Time–temperature data can be accessed using:

```python
curve_data = result["time_temperature"]
```

This separation between the GUI and the calculation kernel makes it possible to integrate the TPS model into other software, automated data-processing workflows, or future standalone applications.

---

## 19. Software Architecture

The program follows a separated GUI–kernel architecture:

```text
┌─────────────────────────────┐
│           main.py           │
│                             │
│     Graphical Interface     │
│ Data Import / Parameters    │
│ Plot / Results / Export     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ TPS_Fitting_CPU_Kernel_ABC  │
│                             │
│ TPS Numerical Kernel        │
│ λ Search                    │
│ Error Evaluation            │
│ Thickness Correction        │
│ Data Output Interface       │
└──────────────┬──────────────┘
               │
               ▼
        λ_hat / λ_corrected
        Fit Error
        Time–Temperature Data
```

This structure keeps the numerical TPS model independent from the graphical interface and facilitates future software development and distribution.

---

## 20. Version Information

**Software:** TPS In-Plane Thermal Conductivity Fitting Software
**Calculation mode:** CPU
**Interface:** Python GUI
**Thickness correction:** A/B/C model
**Default A:** 160372
**Default B:** 218 μm
**Default C:** −0.03 W m⁻¹ K⁻¹

The software is intended for research-oriented analysis of TPS measurements of thin films and membrane materials.
