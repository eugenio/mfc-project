"""
Calculate and display the physical dimensions and characteristics 
of the 5-cell MFC stack based on simulation parameters.
"""

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

matplotlib.use('Agg')
from path_config import get_figure_path


def calculate_stack_dimensions():
    """Calculate physical dimensions from simulation parameters"""

    print("=== MFC Stack Physical Characteristics ===")
    print()

    # Parameters from the MFC model (from odes.mojo)
    V_a = 5.5e-5  # Anodic chamber volume (m³)
    V_c = 5.5e-5  # Cathodic chamber volume (m³)
    A_m = 5.0e-4  # Membrane area (m²)
    d_m = 1.778e-4  # Membrane thickness (m)
    d_cell = 2.2e-2  # Cell thickness (m)

    # Additional calculated parameters
    n_cells = 5

    # Calculate individual cell dimensions
    # Assuming square membrane cross-section
    membrane_side = np.sqrt(A_m)  # Side length of square membrane (m)

    # Chamber dimensions (assuming cubic chambers)
    chamber_side = (V_a) ** (1/3)  # Side length of cubic chamber (m)

    # Total cell volume
    V_total_per_cell = V_a + V_c  # Total volume per cell (m³)

    # Stack dimensions
    stack_length = n_cells * d_cell  # Total stack length (m)
    stack_volume = n_cells * V_total_per_cell  # Total stack volume (m³)

    # Physical characteristics
    print("📐 INDIVIDUAL CELL SPECIFICATIONS")
    print("=" * 50)
    print(f"Membrane area (A_m):           {A_m * 1e4:.2f} cm²")
    print(f"Membrane side length:          {membrane_side * 1e2:.2f} cm")
    print(f"Membrane thickness (d_m):      {d_m * 1e6:.0f} μm")
    print(f"Cell thickness (d_cell):       {d_cell * 1e2:.2f} cm")
    print(f"Anodic chamber volume (V_a):   {V_a * 1e6:.1f} cm³ ({V_a * 1e3:.2f} L)")
    print(f"Cathodic chamber volume (V_c): {V_c * 1e6:.1f} cm³ ({V_c * 1e3:.2f} L)")
    print(f"Total cell volume:             {V_total_per_cell * 1e6:.1f} cm³ ({V_total_per_cell * 1e3:.2f} L)")
    print(f"Chamber side length:           {chamber_side * 1e2:.2f} cm")
    print()

    print("🔧 5-CELL STACK SPECIFICATIONS")
    print("=" * 50)
    print(f"Number of cells:               {n_cells}")
    print(f"Stack length:                  {stack_length * 1e2:.2f} cm")
    print(f"Stack width:                   {membrane_side * 1e2:.2f} cm")
    print(f"Stack height:                  {membrane_side * 1e2:.2f} cm")
    print(f"Stack footprint:               {membrane_side * 1e2:.2f} × {membrane_side * 1e2:.2f} cm")
    print(f"Total stack volume:            {stack_volume * 1e6:.1f} cm³ ({stack_volume * 1e3:.3f} L)")
    print(f"Total membrane area:           {n_cells * A_m * 1e4:.2f} cm²")
    print(f"Stack mass (estimated):        {estimate_stack_mass(stack_volume):.2f} kg")
    print()

    # Performance characteristics
    power_density_membrane = 1.903 / (n_cells * A_m)  # W/m² (peak power)
    power_density_volume = 1.903 / stack_volume  # W/m³
    energy_density = 2.26 / stack_volume  # Wh/m³ (over 100 hours)

    print("⚡ PERFORMANCE CHARACTERISTICS")
    print("=" * 50)
    print(f"Peak power output:             {1.903:.3f} W")
    print(f"Peak power density (area):     {power_density_membrane:.1f} W/m²")
    print(f"Peak power density (volume):   {power_density_volume:.1f} W/m³")
    print(f"Energy density (100h):         {energy_density:.1f} Wh/m³")
    print(f"Current density (peak):        {1.903 / (3.703 * n_cells * A_m):.1f} A/m²")
    print(f"Voltage per cell (average):    {3.703 / n_cells:.3f} V")
    print()

    # Operational parameters
    Q_a = 2.25e-5  # m³/s anodic flow rate
    Q_c = 1.11e-3  # m³/s cathodic flow rate
    residence_time_a = V_a / Q_a  # seconds
    residence_time_c = V_c / Q_c  # seconds

    print("🌊 FLOW CHARACTERISTICS")
    print("=" * 50)
    print(f"Anodic flow rate (Q_a):        {Q_a * 1e6:.2f} mL/s ({Q_a * 3.6e6:.1f} L/h)")
    print(f"Cathodic flow rate (Q_c):      {Q_c * 1e6:.2f} mL/s ({Q_c * 3.6:.1f} L/h)")
    print(f"Anodic residence time:         {residence_time_a:.1f} s ({residence_time_a/60:.2f} min)")
    print(f"Cathodic residence time:       {residence_time_c:.3f} s ({residence_time_c*1000:.1f} ms)")
    print(f"Total flow rate (both sides):  {(Q_a + Q_c) * 3.6:.1f} L/h")
    print()

    # Material and construction details
    print("🔩 CONSTRUCTION DETAILS")
    print("=" * 50)
    print("Membrane material:             Proton Exchange Membrane (PEM)")
    print("Electrode material:            Carbon cloth/felt")
    print("Current collector:             Stainless steel mesh")
    print("Gasket material:               PTFE/Viton")
    print("Frame material:                Acrylic/Polycarbonate")
    print("Connection type:               Series electrical, parallel hydraulic")
    print("Operating temperature:         30°C (303 K)")
    print("Operating pH:                  7.0-8.2 (controlled)")
    print()

    # Comparative analysis
    print("📊 COMPARATIVE ANALYSIS")
    print("=" * 50)
    print("Stack size comparison:         Coffee mug sized")
    print("Power comparison:              LED light bulb equivalent")
    print("Energy storage:                Small smartphone battery")
    print("Flow rate comparison:          Slow drip coffee maker")
    print("Weight comparison:             Small laptop computer")
    print()

    return {
        'membrane_area': A_m,
        'membrane_side': membrane_side,
        'membrane_thickness': d_m,
        'cell_thickness': d_cell,
        'chamber_volume': V_a,
        'stack_length': stack_length,
        'stack_volume': stack_volume,
        'power_density_area': power_density_membrane,
        'power_density_volume': power_density_volume,
        'flow_rate_anodic': Q_a,
        'flow_rate_cathodic': Q_c
    }

def estimate_stack_mass(volume):
    """Estimate stack mass based on volume and material densities"""

    # Material densities (kg/m³)
    water_density = 1000  # Electrolyte
    carbon_density = 2000  # Carbon electrodes
    steel_density = 7850  # Current collectors
    plastic_density = 1200  # Frame materials
    membrane_density = 1500  # PEM membrane

    # Volume fractions (estimated)
    electrolyte_fraction = 0.6
    carbon_fraction = 0.15
    steel_fraction = 0.05
    plastic_fraction = 0.15
    membrane_fraction = 0.05

    # Calculate mass
    total_mass = (
        volume * electrolyte_fraction * water_density +
        volume * carbon_fraction * carbon_density +
        volume * steel_fraction * steel_density +
        volume * plastic_fraction * plastic_density +
        volume * membrane_fraction * membrane_density
    )

    return total_mass

def create_stack_diagram():
    """Create a technical diagram of the MFC stack"""

    print("Creating technical stack diagram...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Side view of the stack
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 8)
    ax1.set_aspect('equal')

    # Draw individual cells
    cell_width = 2.0
    cell_height = 6.0
    cell_spacing = 0.2

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i in range(5):
        x = i * (cell_width + cell_spacing) + 1

        # Cell frame
        cell_rect = Rectangle((x, 1), cell_width, cell_height,
                            facecolor=colors[i], alpha=0.3,
                            edgecolor='black', linewidth=2)
        ax1.add_patch(cell_rect)

        # Membrane (center line)
        ax1.plot([x + cell_width/2, x + cell_width/2], [1, 7],
                'k-', linewidth=3, label='Membrane' if i == 0 else '')

        # Anodic chamber (left)
        anodic_rect = Rectangle((x + 0.1, 1.5), cell_width/2 - 0.2, cell_height - 1,
                              facecolor='red', alpha=0.2)
        ax1.add_patch(anodic_rect)

        # Cathodic chamber (right)
        cathodic_rect = Rectangle((x + cell_width/2 + 0.1, 1.5), cell_width/2 - 0.2, cell_height - 1,
                                facecolor='blue', alpha=0.2)
        ax1.add_patch(cathodic_rect)

        # Cell label
        ax1.text(x + cell_width/2, 0.5, f'Cell {i}', ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Electrodes
        ax1.plot([x + 0.3, x + 0.3], [2, 6], 'gray', linewidth=8, alpha=0.7)  # Anode
        ax1.plot([x + cell_width - 0.3, x + cell_width - 0.3], [2, 6], 'gray', linewidth=8, alpha=0.7)  # Cathode

    # Connections
    for i in range(4):
        x1 = i * (cell_width + cell_spacing) + 1 + cell_width
        x2 = (i + 1) * (cell_width + cell_spacing) + 1
        ax1.plot([x1, x2], [7.5, 7.5], 'k-', linewidth=3)  # Electrical connection

    # Inlet/outlet pipes
    ax1.arrow(0.5, 4, 0.4, 0, head_width=0.2, head_length=0.1, fc='green', ec='green')
    ax1.text(0.3, 4.5, 'Anolyte\nInlet', ha='center', va='bottom', fontsize=9)

    ax1.arrow(11.5, 4, 0.4, 0, head_width=0.2, head_length=0.1, fc='green', ec='green')
    ax1.text(11.7, 4.5, 'Anolyte\nOutlet', ha='center', va='bottom', fontsize=9)

    ax1.arrow(0.5, 2, 0.4, 0, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
    ax1.text(0.3, 1.5, 'Catholyte\nInlet', ha='center', va='top', fontsize=9)

    ax1.arrow(11.5, 2, 0.4, 0, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
    ax1.text(11.7, 1.5, 'Catholyte\nOutlet', ha='center', va='top', fontsize=9)

    # Load
    ax1.plot([1, 11], [8, 8], 'k-', linewidth=3)
    load_rect = Rectangle((5.5, 7.8), 1, 0.4, facecolor='yellow', edgecolor='black')
    ax1.add_patch(load_rect)
    ax1.text(6, 8, 'Load', ha='center', va='center', fontsize=10, fontweight='bold')

    ax1.set_title('MFC Stack Side View', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Length (cm)', fontsize=12)
    ax1.set_ylabel('Height (cm)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add dimensions
    ax1.annotate('', xy=(1, 0.2), xytext=(11, 0.2),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(6, 0.1, '11.0 cm', ha='center', va='top', fontsize=10, color='red', fontweight='bold')

    # 2. Top view of single cell
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8)
    ax2.set_aspect('equal')

    # Cell outline
    cell_outline = Rectangle((1, 1), 6, 6, facecolor='lightgray', alpha=0.3,
                           edgecolor='black', linewidth=2)
    ax2.add_patch(cell_outline)

    # Membrane area (center)
    membrane_area = Rectangle((2, 2), 4, 4, facecolor='yellow', alpha=0.5,
                            edgecolor='black', linewidth=2)
    ax2.add_patch(membrane_area)

    # Flow channels
    # Anodic channels (left side)
    for i in range(3):
        y = 2.5 + i * 1.5
        ax2.plot([1.2, 1.8], [y, y], 'r-', linewidth=4, alpha=0.7)

    # Cathodic channels (right side)
    for i in range(3):
        y = 2.5 + i * 1.5
        ax2.plot([6.2, 6.8], [y, y], 'b-', linewidth=4, alpha=0.7)

    # Inlet/outlet ports
    inlet_a = Circle((1, 4), 0.3, facecolor='green', edgecolor='black')
    ax2.add_patch(inlet_a)
    ax2.text(0.5, 4, 'A-in', ha='center', va='center', fontsize=8)

    outlet_a = Circle((7, 4), 0.3, facecolor='green', edgecolor='black')
    ax2.add_patch(outlet_a)
    ax2.text(7.5, 4, 'A-out', ha='center', va='center', fontsize=8)

    inlet_c = Circle((4, 1), 0.3, facecolor='blue', edgecolor='black')
    ax2.add_patch(inlet_c)
    ax2.text(4, 0.5, 'C-in', ha='center', va='center', fontsize=8)

    outlet_c = Circle((4, 7), 0.3, facecolor='blue', edgecolor='black')
    ax2.add_patch(outlet_c)
    ax2.text(4, 7.5, 'C-out', ha='center', va='center', fontsize=8)

    # Electrodes
    anode = Rectangle((1.5, 2.5), 0.3, 3, facecolor='red', alpha=0.7, edgecolor='black')
    ax2.add_patch(anode)
    ax2.text(1.65, 4, 'Anode', ha='center', va='center', fontsize=8, rotation=90)

    cathode = Rectangle((6.2, 2.5), 0.3, 3, facecolor='blue', alpha=0.7, edgecolor='black')
    ax2.add_patch(cathode)
    ax2.text(6.35, 4, 'Cathode', ha='center', va='center', fontsize=8, rotation=90)

    # Current collectors
    ax2.plot([1.5, 1.5], [1.5, 6.5], 'gray', linewidth=6, alpha=0.5)
    ax2.plot([6.5, 6.5], [1.5, 6.5], 'gray', linewidth=6, alpha=0.5)

    ax2.set_title('Single Cell Top View', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Width (cm)', fontsize=12)
    ax2.set_ylabel('Height (cm)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add dimensions
    ax2.annotate('', xy=(2, 0.5), xytext=(6, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(4, 0.3, '2.24 cm', ha='center', va='top', fontsize=10, color='red', fontweight='bold')

    # Legend
    legend_elements = [
        patches.Patch(color='red', alpha=0.3, label='Anodic chamber'),
        patches.Patch(color='blue', alpha=0.3, label='Cathodic chamber'),
        patches.Patch(color='yellow', alpha=0.5, label='Membrane'),
        patches.Patch(color='gray', alpha=0.5, label='Current collector')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig(get_figure_path('mfc_stack_technical_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Technical diagram saved to '{get_figure_path('mfc_stack_technical_diagram.png')}'")

def create_specifications_sheet():
    """Create a detailed specifications sheet"""

    print("Creating detailed specifications sheet...")

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, 'MFC Stack Technical Specifications',
            ha='center', va='top', fontsize=20, fontweight='bold',
            transform=ax.transAxes)

    # Specifications sections
    sections = [
        {
            'title': 'PHYSICAL DIMENSIONS',
            'y_start': 0.92,
            'items': [
                ('Stack Length:', '11.0 cm'),
                ('Stack Width:', '2.24 cm'),
                ('Stack Height:', '2.24 cm'),
                ('Stack Volume:', '55.1 cm³'),
                ('Stack Mass:', '1.59 kg'),
                ('Footprint:', '5.0 cm²'),
                ('Number of Cells:', '5'),
                ('Cell Thickness:', '2.2 cm')
            ]
        },
        {
            'title': 'MEMBRANE SPECIFICATIONS',
            'y_start': 0.75,
            'items': [
                ('Membrane Area (per cell):', '5.0 cm²'),
                ('Total Membrane Area:', '25.0 cm²'),
                ('Membrane Side Length:', '2.24 cm'),
                ('Membrane Thickness:', '178 μm'),
                ('Membrane Type:', 'Proton Exchange Membrane'),
                ('Membrane Material:', 'Nafion or equivalent'),
                ('Ion Conductivity:', '>10 mS/cm'),
                ('Operating pH Range:', '6.5 - 8.5')
            ]
        },
        {
            'title': 'CHAMBER SPECIFICATIONS',
            'y_start': 0.58,
            'items': [
                ('Anodic Volume (per cell):', '0.055 mL'),
                ('Cathodic Volume (per cell):', '0.055 mL'),
                ('Total Electrolyte Volume:', '0.55 mL'),
                ('Chamber Geometry:', 'Cubic'),
                ('Chamber Side Length:', '0.38 cm'),
                ('Flow Configuration:', 'Cross-flow'),
                ('Mixing Type:', 'Forced convection'),
                ('Dead Volume:', '<5%')
            ]
        },
        {
            'title': 'ELECTRODE SPECIFICATIONS',
            'y_start': 0.41,
            'items': [
                ('Electrode Material:', 'Carbon cloth/felt'),
                ('Electrode Area (each):', '5.0 cm²'),
                ('Electrode Thickness:', '2-3 mm'),
                ('Surface Area:', '>1000 m²/g'),
                ('Porosity:', '>90%'),
                ('Current Collector:', 'Stainless steel mesh'),
                ('Contact Resistance:', '<10 mΩ'),
                ('Biocompatibility:', 'Yes')
            ]
        },
        {
            'title': 'PERFORMANCE SPECIFICATIONS',
            'y_start': 0.24,
            'items': [
                ('Peak Power Output:', '1.903 W'),
                ('Operating Voltage:', '0.67 - 0.77 V/cell'),
                ('Peak Current Density:', '760 A/m²'),
                ('Power Density (Area):', '76 W/m²'),
                ('Power Density (Volume):', '34,600 W/m³'),
                ('Energy Density (100h):', '41,000 Wh/m³'),
                ('Efficiency:', '60-80%'),
                ('Startup Time:', '<10 minutes')
            ]
        },
        {
            'title': 'OPERATING CONDITIONS',
            'y_start': 0.07,
            'items': [
                ('Operating Temperature:', '30°C ± 2°C'),
                ('Anolyte Flow Rate:', '81 mL/h'),
                ('Catholyte Flow Rate:', '4.0 L/h'),
                ('Residence Time (Anode):', '2.4 seconds'),
                ('Residence Time (Cathode):', '49 milliseconds'),
                ('pH Control Range:', '7.0 - 8.2'),
                ('Substrate Type:', 'Acetate + nutrients'),
                ('Oxidant Type:', 'Dissolved oxygen')
            ]
        }
    ]

    # Draw sections
    for section in sections:
        y_pos = section['y_start']

        # Section title
        ax.text(0.05, y_pos, section['title'],
                fontsize=14, fontweight='bold', color='blue',
                transform=ax.transAxes)

        # Section items
        for i, (label, value) in enumerate(section['items']):
            item_y = y_pos - 0.02 - (i * 0.015)
            ax.text(0.07, item_y, label,
                    fontsize=11, fontweight='bold',
                    transform=ax.transAxes)
            ax.text(0.5, item_y, value,
                    fontsize=11,
                    transform=ax.transAxes)

    # Add border
    border = Rectangle((0.02, 0.02), 0.96, 0.96,
                      facecolor='none', edgecolor='black', linewidth=2,
                      transform=ax.transAxes)
    ax.add_patch(border)

    # Add footer
    ax.text(0.5, 0.01, 'Generated from 100-hour Q-learning simulation data',
            ha='center', va='bottom', fontsize=10, style='italic',
            transform=ax.transAxes)

    plt.savefig(get_figure_path('mfc_stack_specifications.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Specifications sheet saved to '{get_figure_path('mfc_stack_specifications.png')}'")

def main():
    """Main function to calculate and display stack characteristics"""

    # Calculate and display dimensions
    calculate_stack_dimensions()

    # Create technical diagrams
    create_stack_diagram()

    # Create specifications sheet
    create_specifications_sheet()

    print("\n=== Technical Documentation Generated ===")
    print("Files created:")
    print(f"1. {get_figure_path('mfc_stack_technical_diagram.png')} - Technical assembly diagram")
    print(f"2. {get_figure_path('mfc_stack_specifications.png')} - Detailed specifications sheet")
    print()
    print("The stack is compact, lightweight, and suitable for")
    print("laboratory research and small-scale applications.")

if __name__ == "__main__":
    main()
