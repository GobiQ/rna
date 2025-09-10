import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
import seaborn as sns
from io import StringIO
import re
import base64
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Set page config
st.set_page_config(
    page_title="RNA Structure Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .sequence-display {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        background-color: #ffffff;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #bdc3c7;
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.4;
    }
    .structure-display {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        background-color: #ffffff;
        color: #27ae60;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #27ae60;
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.4;
    }
    .energy-box {
        background-color: #ffffff;
        color: #2c3e50;
        border: 2px solid #f39c12;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .energy-box h4 {
        color: #e67e22;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class StructuralElement:
    """Class to represent a structural element in RNA"""
    type: str  # 'hairpin', 'internal_loop', 'bulge', 'stem', 'multibranch'
    start: int
    end: int
    loop_size: Optional[int] = None
    sequence: Optional[str] = None
    paired_with: Optional[int] = None
    stem_length: Optional[int] = None

class RNA2DLayoutEngine:
    """Engine for creating 2D RNA structure layouts"""
    
    def __init__(self, sequence: str, structure: str):
        self.sequence = sequence
        self.structure = structure
        self.length = len(sequence)
        self.positions = {}  # Will store x, y coordinates for each nucleotide
        self.elements = []
        self.base_pairs = self._find_base_pairs()
        
    def _find_base_pairs(self) -> List[Tuple[int, int]]:
        """Find all base pairs from dot-bracket notation"""
        pairs = []
        stack = []
        
        for i, char in enumerate(self.structure):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                j = stack.pop()
                pairs.append((j, i))
        
        return pairs
    
    def _analyze_structure_elements(self) -> List[StructuralElement]:
        """Analyze and categorize structural elements"""
        elements = []
        
        # Find stems and loops
        for start, end in self.base_pairs:
            # Check for hairpin loops
            loop_size = end - start - 1
            if loop_size > 0:
                loop_seq = self.sequence[start+1:end]
                
                # Determine loop type
                if loop_size <= 8:  # Typical hairpin
                    elements.append(StructuralElement(
                        type='hairpin',
                        start=start,
                        end=end,
                        loop_size=loop_size,
                        sequence=loop_seq
                    ))
                else:  # Larger internal loop
                    elements.append(StructuralElement(
                        type='internal_loop',
                        start=start,
                        end=end,
                        loop_size=loop_size,
                        sequence=loop_seq
                    ))
        
        return elements
    
    def calculate_radial_layout(self) -> Dict[int, Tuple[float, float]]:
        """Calculate positions using a radial layout algorithm"""
        positions = {}
        
        if not self.base_pairs:
            # Linear layout for unpaired sequence
            for i in range(self.length):
                positions[i] = (i * 0.8, 0)
            return positions
        
        # Find the outermost loop
        center_x, center_y = 0, 0
        radius = max(10, self.length * 0.15)
        
        # Place nucleotides in a circular arrangement
        angle_step = 2 * np.pi / self.length
        
        for i in range(self.length):
            angle = i * angle_step
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            positions[i] = (x, y)
        
        # Adjust positions for base pairs to bring them closer
        for start, end in self.base_pairs:
            start_pos = positions[start]
            end_pos = positions[end]
            
            # Move paired bases closer together
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            
            # Reduce distance between paired bases
            factor = 0.7
            new_start_x = mid_x + (start_pos[0] - mid_x) * factor
            new_start_y = mid_y + (start_pos[1] - mid_y) * factor
            new_end_x = mid_x + (end_pos[0] - mid_x) * factor
            new_end_y = mid_y + (end_pos[1] - mid_y) * factor
            
            positions[start] = (new_start_x, new_start_y)
            positions[end] = (new_end_x, new_end_y)
        
        return positions
    
    def calculate_hierarchical_layout(self) -> Dict[int, Tuple[float, float]]:
        """Calculate positions using hierarchical layout for complex structures"""
        positions = {}
        
        # Identify nested structure levels
        depth_map = [0] * self.length
        current_depth = 0
        
        for i, char in enumerate(self.structure):
            if char == '(':
                current_depth += 1
            elif char == ')':
                current_depth -= 1
            depth_map[i] = current_depth
        
        max_depth = max(depth_map) if depth_map else 0
        
        # Arrange by depth levels
        y_spacing = 1.5
        x_spacing = 1.0
        
        for i in range(self.length):
            depth = depth_map[i]
            x = i * x_spacing
            y = depth * y_spacing
            positions[i] = (x, y)
        
        return positions

class RNAStructurePredictor:
    """Enhanced RNA secondary structure predictor"""
    
    def __init__(self):
        # Base pairing rules with improved energies
        self.base_pairs = {
            ('A', 'U'): -2.0, ('U', 'A'): -2.0,
            ('G', 'C'): -3.0, ('C', 'G'): -3.0,
            ('G', 'U'): -1.0, ('U', 'G'): -1.0
        }
        
        # Stacking energies (simplified)
        self.stacking_energies = {
            'AU': -1.1, 'UA': -1.3, 'GC': -2.4, 'CG': -2.1,
            'GU': -0.6, 'UG': -1.4
        }
        
        # Loop penalties
        self.loop_penalties = {
            'hairpin': {3: 5.7, 4: 5.6, 5: 5.6, 6: 5.4, 7: 5.9, 8: 6.4, 9: 6.9},
            'bulge': 3.8,
            'internal': 4.1
        }
    
    def is_valid_rna(self, sequence):
        """Validate RNA sequence (allows T for DNA sequences)"""
        return bool(re.match(r'^[AUTGC]+$', sequence.upper()))
    
    def get_invalid_characters(self, sequence):
        """Get list of invalid characters in sequence"""
        valid_chars = set('AUTGC')
        return [char for char in set(sequence.upper()) if char not in valid_chars]
    
    def can_pair(self, base1, base2):
        """Check if two bases can pair"""
        return (base1, base2) in self.base_pairs
    
    def get_pairing_energy(self, base1, base2):
        """Get energy for base pairing"""
        return self.base_pairs.get((base1, base2), 0)
    
    def simple_fold(self, sequence):
        """Enhanced folding algorithm with better energy estimation"""
        n = len(sequence)
        if n < 4:
            return '.' * n, 0.0
        
        # DP table for maximum number of base pairs
        dp = [[0 for _ in range(n)] for _ in range(n)]
        traceback = [[None for _ in range(n)] for _ in range(n)]
        
        # Fill DP table with improved scoring
        for length in range(4, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Option 1: don't pair i and j
                if j > i:
                    dp[i][j] = dp[i][j-1]
                
                # Option 2: pair i and j (if possible)
                if self.can_pair(sequence[i], sequence[j]) and j - i >= 3:
                    # Base pairing energy bonus
                    energy_gain = abs(self.get_pairing_energy(sequence[i], sequence[j]))
                    
                    # Add stacking bonus if possible
                    stacking_bonus = 0
                    if i > 0 and j < n-1 and self.can_pair(sequence[i-1], sequence[j+1]):
                        pair_type = sequence[i] + sequence[j]
                        stacking_bonus = self.stacking_energies.get(pair_type, 0)
                    
                    total_score = dp[i+1][j-1] + energy_gain + abs(stacking_bonus)
                    
                    if total_score > dp[i][j]:
                        dp[i][j] = total_score
                        traceback[i][j] = 'pair'
                
                # Option 3: split into two subproblems
                for k in range(i, j):
                    if dp[i][k] + dp[k+1][j] > dp[i][j]:
                        dp[i][j] = dp[i][k] + dp[k+1][j]
                        traceback[i][j] = k
        
        # Traceback to get structure
        structure = ['.'] * n
        self._traceback(sequence, traceback, structure, 0, n-1)
        
        # Calculate free energy with loop penalties
        free_energy = self._calculate_detailed_energy(sequence, ''.join(structure))
        
        return ''.join(structure), free_energy
    
    def _traceback(self, sequence, traceback, structure, i, j):
        """Recursive traceback to determine structure"""
        if i >= j:
            return
        
        if traceback[i][j] == 'pair':
            structure[i] = '('
            structure[j] = ')'
            self._traceback(sequence, traceback, structure, i+1, j-1)
        elif isinstance(traceback[i][j], int):
            k = traceback[i][j]
            self._traceback(sequence, traceback, structure, i, k)
            self._traceback(sequence, traceback, structure, k+1, j)
        else:
            self._traceback(sequence, traceback, structure, i, j-1)
    
    def _calculate_detailed_energy(self, sequence, structure):
        """Calculate detailed free energy with loop penalties"""
        energy = 0.0
        stack = []
        
        # Base pairing energies
        for i, symbol in enumerate(structure):
            if symbol == '(':
                stack.append(i)
            elif symbol == ')' and stack:
                j = stack.pop()
                energy += self.get_pairing_energy(sequence[j], sequence[i])
                
                # Add loop penalty
                loop_size = i - j - 1
                if loop_size > 0:
                    if loop_size <= 9:
                        energy += self.loop_penalties['hairpin'].get(loop_size, 7.0)
                    else:
                        energy += 7.0 + 1.75 * np.log(loop_size / 9.0)
        
        return energy

def create_2d_structure_plot(sequence: str, structure: str, energy: float):
    """Create enhanced 2D structure visualization like the reference image"""
    
    layout_engine = RNA2DLayoutEngine(sequence, structure)
    
    # Calculate positions using hierarchical layout for better structure representation
    positions = layout_engine.calculate_hierarchical_layout()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Hierarchical layout
    _plot_structure_layout(ax1, sequence, structure, positions, "Hierarchical Layout", energy)
    
    # Plot 2: Radial layout for comparison
    radial_positions = layout_engine.calculate_radial_layout()
    _plot_structure_layout(ax2, sequence, structure, radial_positions, "Radial Layout", energy)
    
    plt.tight_layout()
    return fig

def _plot_structure_layout(ax, sequence: str, structure: str, positions: Dict[int, Tuple[float, float]], title: str, energy: float):
    """Helper function to plot RNA structure layout"""
    
    # Base colors
    base_colors = {
        'A': '#e74c3c',  # Red
        'U': '#3498db',  # Blue  
        'G': '#2ecc71',  # Green
        'C': '#f39c12',  # Orange
        'T': '#3498db'   # Blue (same as U)
    }
    
    # Plot nucleotides
    for i, (x, y) in positions.items():
        base = sequence[i]
        color = base_colors.get(base, '#95a5a6')
        
        # Draw nucleotide circle
        circle = Circle((x, y), 0.3, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
        
        # Add base label
        ax.text(x, y, base, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Add position number (for reference)
        ax.text(x, y-0.6, str(i+1), ha='center', va='center', fontsize=8, color='gray')
    
    # Draw base pairs
    stack = []
    pair_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    color_idx = 0
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            
            # Get positions
            pos1 = positions[j]
            pos2 = positions[i]
            
            # Draw base pair line
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                   color=pair_colors[color_idx % len(pair_colors)], 
                   linewidth=3, alpha=0.7, zorder=1)
            
            # Draw Watson-Crick symbols
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            
            base1, base2 = sequence[j], sequence[i]
            if (base1, base2) in [('G', 'C'), ('C', 'G')]:
                # Triple bond for GC
                for offset in [-0.1, 0, 0.1]:
                    ax.plot([pos1[0], pos2[0]], [pos1[1] + offset, pos2[1] + offset], 
                           'k-', linewidth=1, alpha=0.5)
            elif (base1, base2) in [('A', 'U'), ('U', 'A'), ('A', 'T'), ('T', 'A')]:
                # Double bond for AU/AT
                for offset in [-0.05, 0.05]:
                    ax.plot([pos1[0], pos2[0]], [pos1[1] + offset, pos2[1] + offset], 
                           'k-', linewidth=1, alpha=0.5)
            elif (base1, base2) in [('G', 'U'), ('U', 'G')]:
                # Wobble pair - dashed line
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'k--', linewidth=2, alpha=0.5)
            
            color_idx += 1
    
    # Identify and label structural elements
    _label_structural_elements(ax, sequence, structure, positions)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nΔG ≈ {energy:.2f} kcal/mol', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='A'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='U/T'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='G'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markersize=10, label='C'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Base Pairs'),
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Wobble Pairs')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

def _label_structural_elements(ax, sequence: str, structure: str, positions: Dict[int, Tuple[float, float]]):
    """Add labels for structural elements like loops, stems, etc."""
    
    # Find hairpin loops
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            start = stack.pop()
            loop_size = i - start - 1
            
            if loop_size > 0 and loop_size <= 8:  # Hairpin loop
                # Calculate center of loop
                loop_positions = [positions[j] for j in range(start + 1, i)]
                if loop_positions:
                    center_x = np.mean([pos[0] for pos in loop_positions])
                    center_y = np.mean([pos[1] for pos in loop_positions])
                    
                    # Add hairpin label
                    ax.annotate('hairpin loop', 
                              xy=(center_x, center_y), 
                              xytext=(center_x + 1, center_y + 1),
                              arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                              fontsize=10, color='blue', fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            
            elif loop_size > 8:  # Internal loop
                loop_positions = [positions[j] for j in range(start + 1, i)]
                if loop_positions:
                    center_x = np.mean([pos[0] for pos in loop_positions])
                    center_y = np.mean([pos[1] for pos in loop_positions])
                    
                    # Add internal loop label
                    ax.annotate('internal loop', 
                              xy=(center_x, center_y), 
                              xytext=(center_x + 1, center_y - 1),
                              arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                              fontsize=10, color='purple', fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', alpha=0.7))

def analyze_sequence(sequence):
    """Analyze RNA sequence composition"""
    sequence_upper = sequence.upper()
    composition = {base: sequence_upper.count(base) for base in 'AUGC'}
    
    # Handle T as well
    if 'T' in sequence_upper:
        composition['T'] = sequence_upper.count('T')
    
    total_bases = sum(composition.values())
    gc_content = (composition.get('G', 0) + composition.get('C', 0)) / total_bases * 100 if total_bases > 0 else 0
    
    return composition, gc_content

def create_comprehensive_analysis_plot(sequence, structure, energy):
    """Create comprehensive analysis including composition, energy landscape, etc."""
    fig = plt.figure(figsize=(20, 15))
    
    # Create complex subplot layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1.5, 1], width_ratios=[1, 1, 1])
    
    # Composition analysis
    ax1 = fig.add_subplot(gs[0, 0])
    composition, gc_content = analyze_sequence(sequence)
    
    bases = list(composition.keys())
    counts = list(composition.values())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'][:len(bases)]
    
    bars = ax1.bar(bases, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_title('Base Composition', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # GC content pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    gc_count = composition.get('G', 0) + composition.get('C', 0)
    at_count = len(sequence) - gc_count
    
    wedges, texts, autotexts = ax2.pie([gc_count, at_count], 
                                      labels=['GC', 'AU/T'], 
                                      colors=['#27ae60', '#e74c3c'],
                                      autopct='%1.1f%%',
                                      startangle=90)
    ax2.set_title('GC Content', fontsize=12, fontweight='bold')
    
    # Structure statistics
    ax3 = fig.add_subplot(gs[0, 2])
    base_pairs = structure.count('(')
    unpaired = structure.count('.')
    
    stats_data = [base_pairs, unpaired]
    stats_labels = ['Paired', 'Unpaired']
    
    bars = ax3.bar(stats_labels, stats_data, color=['#3498db', '#95a5a6'], alpha=0.8)
    ax3.set_title('Pairing Statistics', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Nucleotides')
    
    for bar, count in zip(bars, stats_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Energy landscape
    ax4 = fig.add_subplot(gs[1, :])
    _plot_energy_landscape(ax4, sequence, structure, energy)
    
    # Main 2D structure (spanning multiple cells)
    ax5 = fig.add_subplot(gs[2, :])
    layout_engine = RNA2DLayoutEngine(sequence, structure)
    positions = layout_engine.calculate_hierarchical_layout()
    _plot_structure_layout(ax5, sequence, structure, positions, "2D Secondary Structure Diagram", energy)
    
    # Structural elements analysis
    ax6 = fig.add_subplot(gs[3, :])
    _plot_structural_elements_summary(ax6, sequence, structure)
    
    plt.tight_layout()
    return fig

def _plot_energy_landscape(ax, sequence, structure, energy):
    """Plot energy landscape along the sequence"""
    # Simplified energy profile
    positions = range(len(sequence))
    energies = []
    
    current_energy = 0
    stack = []
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
            current_energy += 0.5  # Opening penalty
        elif char == ')' and stack:
            start = stack.pop()
            # Base pairing energy
            predictor = RNAStructurePredictor()
            pair_energy = predictor.get_pairing_energy(sequence[start], sequence[i])
            current_energy += pair_energy
        else:
            current_energy += 0.1  # Loop penalty
        
        energies.append(current_energy)
    
    ax.plot(positions, energies, 'b-', linewidth=2, alpha=0.8)
    ax.fill_between(positions, energies, alpha=0.3, color='lightblue')
    ax.set_title('Energy Landscape Along Sequence', fontsize=12, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Cumulative Energy (kcal/mol)')
    ax.grid(True, alpha=0.3)
    
    # Highlight base pairs
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            start = stack.pop()
            ax.axvspan(start, i, alpha=0.2, color='green')

def _plot_structural_elements_summary(ax, sequence, structure):
    """Plot summary of structural elements"""
    
    # Analyze structural elements
    elements = {
        'Hairpins': 0,
        'Internal Loops': 0,
        'Bulges': 0,
        'Stems': 0
    }
    
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            start = stack.pop()
            loop_size = i - start - 1
            
            elements['Stems'] += 1
            
            if loop_size == 0:
                continue
            elif loop_size <= 8:
                elements['Hairpins'] += 1
            elif loop_size <= 2:
                elements['Bulges'] += 1
            else:
                elements['Internal Loops'] += 1
    
    # Create bar plot
    element_names = list(elements.keys())
    element_counts = list(elements.values())
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db']
    
    bars = ax.bar(element_names, element_counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_title('Structural Elements Summary', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, element_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}', ha='center', va='bottom', fontweight='bold')

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Enhanced RNA Structure Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This enhanced application predicts RNA secondary structure and creates detailed 2D visualizations 
    similar to professional RNA structure diagrams. Features include labeled structural elements, 
    multiple layout algorithms, and comprehensive analysis.
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Options")
        
        # Example sequences
        st.markdown("#### Example Sequences")
        examples = {
            "tRNA-like": "GCGCAAUUAGGCGCGUCCCUCCACCCUGUGCCUUUCCAGGGCUGGGCAAGAUUCUGCGAACGACUCCCGCCGUGUUU",
            "Complex Hairpin": "GGCUUUAGCCUUCGCCACCAUGAGCGUGGUACCUCCGAGCUUCGAGCGGCCCC",
            "Simple Hairpin": "GGGGAAAACCCC",
            "Multi-branch Structure": "CUGCUGUCAGCCGGACUACUUGUGCGCGCAAUAACCCUAGGGCUGCCUUCGGGAACCUGUCUGUAA",
            "Pseudoknot Example": "GCGCAACGCGCUUUUCGCGUUGCGC",
            "Multiple Hairpins": "GGCUUUAGCCCCAAAAGGGGUUUUCCCC",
            "Ribosomal Fragment": "GGGCGAUGAGGCCCGCCCAAACGACCCCCCUGUGAUGUUCGCAGAAGCGCGGCAGCGCGCCUGCUAUUGGGCGCGGCCCCGGGCGGGCGCCCCCGCGGGAGAGCCGGCGGGAAACCCGGUCCCAUCCCCGAGGCGGGGGCAGGCGCCACCCCCCGCCGAUCCCCGAGCCGGCGGGCCCGAUCCCCGAGCCGGCGGGCCCGAUCCCCGAGCCCGCCCGGGAGAGCCGGCG"
        }
        
        selected_example = st.selectbox("Choose an example:", ["None"] + list(examples.keys()))
        
        # Layout options
        st.markdown("#### Visualization Options")
        layout_type = st.selectbox("Layout Algorithm:", ["Hierarchical", "Radial", "Both"])
        show_labels = st.checkbox("Show structural element labels", value=True)
        show_energy_profile = st.checkbox("Show energy landscape", value=True)
        color_by_structure = st.checkbox("Color by structure depth", value=False)
        
        # Input options
        st.markdown("#### Input Options")
        convert_t_to_u = st.checkbox("Convert T to U", value=True, help="Convert DNA sequence to RNA")
        remove_spaces = st.checkbox("Remove spaces", value=True)
        uppercase = st.checkbox("Convert to uppercase", value=True)
    
    # Main input
    st.markdown('<div class="section-header">Input RNA Sequence</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if selected_example != "None":
            default_seq = examples[selected_example]
        else:
            default_seq = ""
        
        # Dynamic label based on conversion settings
        if convert_t_to_u:
            sequence_label = "Enter sequence (DNA or RNA):"
        else:
            sequence_label = "Enter RNA sequence (A, U, G, C only):"
            
        sequence_input = st.text_area(
            sequence_label,
            value=default_seq,
            height=100,
            placeholder="GGCUUUAGCCUUCGCCACCAUGAGCG..."
        )
    
    with col2:
        st.markdown("**Current Settings:**")
        st.markdown(f"• Convert T to U: {'✅' if convert_t_to_u else '❌'}")
        st.markdown(f"• Remove spaces: {'✅' if remove_spaces else '❌'}")
        st.markdown(f"• Uppercase: {'✅' if uppercase else '❌'}")
        st.markdown(f"• Layout: {layout_type}")
        st.markdown(f"• Show labels: {'✅' if show_labels else '❌'}")
    
    # Process sequence
    if sequence_input:
        processed_seq = sequence_input.strip()
        
        if remove_spaces:
            processed_seq = re.sub(r'\s+', '', processed_seq)
        if convert_t_to_u:
            processed_seq = processed_seq.replace('T', 'U')
        if uppercase:
            processed_seq = processed_seq.upper()
        
        # Validate sequence
        predictor = RNAStructurePredictor()
        
        if predictor.is_valid_rna(processed_seq):
            st.success(f"✅ Valid RNA sequence of length {len(processed_seq)}")
            
            # Display processed sequence
            st.markdown('<div class="section-header">Processed Sequence</div>', unsafe_allow_html=True)
            
            # Split long sequences into lines of 60 characters for better display
            formatted_seq = ""
            for i in range(0, len(processed_seq), 60):
                line_start = i + 1
                line_end = min(i + 60, len(processed_seq))
                line = processed_seq[i:i+60]
                formatted_seq += f"{line_start:>5}: {line}\n"
            
            st.markdown(f'<div class="sequence-display">{formatted_seq}</div>', unsafe_allow_html=True)
            
            # Sequence analysis
            composition, gc_content = analyze_sequence(processed_seq)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Length", len(processed_seq))
            with col2:
                st.metric("GC Content", f"{gc_content:.1f}%")
            with col3:
                st.metric("Base Pairs", "Calculating...")
            with col4:
                st.metric("Structure Complexity", "Calculating...")
            
            # Structure prediction
            with st.spinner("Predicting RNA structure using enhanced algorithm..."):
                structure, energy = predictor.simple_fold(processed_seq)
            
            # Calculate additional metrics
            base_pairs = structure.count('(')
            max_depth = 0
            current_depth = 0
            for char in structure:
                if char == '(':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == ')':
                    current_depth -= 1
            
            # Update metrics
            col3.metric("Base Pairs", base_pairs)
            col4.metric("Max Nesting Depth", max_depth)
            
            # Display results
            st.markdown('<div class="section-header">Structure Prediction Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sequence:**")
                st.markdown(f'<div class="sequence-display">{formatted_seq}</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown("**Structure (Dot-Bracket):**")
                # Format structure display
                formatted_struct = ""
                for i in range(0, len(structure), 60):
                    line_start = i + 1
                    line = structure[i:i+60]
                    formatted_struct += f"{line_start:>5}: {line}\n"
                st.markdown(f'<div class="structure-display">{formatted_struct}</div>', unsafe_allow_html=True)
            
            # Enhanced energy information
            stability = "High" if energy < -10 else "Medium" if energy < -5 else "Low"
            pairing_efficiency = (base_pairs * 2 / len(processed_seq) * 100)
            
            st.markdown(f'''
            <div class="energy-box">
                <h4>🔬 Thermodynamic Analysis</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <p><strong>Estimated Free Energy (ΔG):</strong> {energy:.2f} kcal/mol</p>
                        <p><strong>Structure Stability:</strong> {stability}</p>
                        <p><strong>Base Pairs:</strong> {base_pairs}</p>
                    </div>
                    <div>
                        <p><strong>Pairing Efficiency:</strong> {pairing_efficiency:.1f}%</p>
                        <p><strong>Maximum Nesting:</strong> {max_depth}</p>
                        <p><strong>Unpaired Bases:</strong> {structure.count('.')}</p>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Enhanced 2D Structure Visualization
            st.markdown('<div class="section-header">🎨 Enhanced 2D Structure Diagrams</div>', unsafe_allow_html=True)
            
            if layout_type == "Both":
                st.markdown("**Comparison of Layout Algorithms:**")
                fig = create_2d_structure_plot(processed_seq, structure, energy)
                st.pyplot(fig)
            else:
                # Single layout
                layout_engine = RNA2DLayoutEngine(processed_seq, structure)
                
                if layout_type == "Hierarchical":
                    positions = layout_engine.calculate_hierarchical_layout()
                else:  # Radial
                    positions = layout_engine.calculate_radial_layout()
                
                fig, ax = plt.subplots(1, 1, figsize=(15, 12))
                _plot_structure_layout(ax, processed_seq, structure, positions, f"{layout_type} Layout", energy)
                
                # Enhance with labels if requested
                if show_labels:
                    _label_structural_elements(ax, processed_seq, structure, positions)
                
                st.pyplot(fig)
            
            # Comprehensive Analysis
            if show_energy_profile:
                st.markdown('<div class="section-header">📊 Comprehensive Structure Analysis</div>', unsafe_allow_html=True)
                
                comp_fig = create_comprehensive_analysis_plot(processed_seq, structure, energy)
                st.pyplot(comp_fig)
            
            # Detailed structural analysis
            st.markdown('<div class="section-header">🔍 Structural Elements Breakdown</div>', unsafe_allow_html=True)
            
            # Analyze structural elements in detail
            layout_engine = RNA2DLayoutEngine(processed_seq, structure)
            elements = layout_engine._analyze_structure_elements()
            
            # Categorize elements
            hairpins = [e for e in elements if e.type == 'hairpin']
            internal_loops = [e for e in elements if e.type == 'internal_loop']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🎀 Hairpin Loops")
                if hairpins:
                    for i, hairpin in enumerate(hairpins):
                        with st.expander(f"Hairpin {i+1} (positions {hairpin.start+1}-{hairpin.end+1})"):
                            st.markdown(f"**Loop sequence:** `{hairpin.sequence}`")
                            st.markdown(f"**Loop size:** {hairpin.loop_size} nucleotides")
                            st.markdown(f"**Stability:** {'High' if hairpin.loop_size in [4, 5, 6] else 'Medium'}")
                            
                            # Analyze loop sequence
                            gc_in_loop = hairpin.sequence.count('G') + hairpin.sequence.count('C')
                            st.markdown(f"**GC content in loop:** {gc_in_loop}/{hairpin.loop_size}")
                else:
                    st.info("No hairpin loops detected")
            
            with col2:
                st.markdown("#### 🔄 Internal Loops & Bulges")
                if internal_loops:
                    for i, loop in enumerate(internal_loops):
                        loop_type = "Bulge" if loop.loop_size <= 3 else "Internal Loop"
                        with st.expander(f"{loop_type} {i+1} (positions {loop.start+1}-{loop.end+1})"):
                            st.markdown(f"**Sequence:** `{loop.sequence}`")
                            st.markdown(f"**Size:** {loop.loop_size} nucleotides")
                            st.markdown(f"**Type:** {loop_type}")
                            
                            if loop.loop_size > 10:
                                st.warning("⚠️ Large loop - may affect structure stability")
                else:
                    st.info("No internal loops or bulges detected")
            
            with col3:
                st.markdown("#### 🔗 Base Pairing Analysis")
                
                # Analyze all base pairs
                stack = []
                pair_analysis = {'Watson-Crick': [], 'Wobble': []}
                
                for i, char in enumerate(structure):
                    if char == '(':
                        stack.append(i)
                    elif char == ')' and stack:
                        start = stack.pop()
                        base1, base2 = processed_seq[start], processed_seq[i]
                        pair_type = f"{base1}-{base2}"
                        
                        if (base1, base2) in [('G','C'), ('C','G'), ('A','U'), ('U','A')]:
                            pair_analysis['Watson-Crick'].append((start+1, i+1, pair_type))
                        else:
                            pair_analysis['Wobble'].append((start+1, i+1, pair_type))
                
                st.markdown(f"**Total base pairs:** {len(pair_analysis['Watson-Crick']) + len(pair_analysis['Wobble'])}")
                st.markdown(f"**Watson-Crick pairs:** {len(pair_analysis['Watson-Crick'])}")
                st.markdown(f"**Wobble pairs:** {len(pair_analysis['Wobble'])}")
                
                if pair_analysis['Watson-Crick'] or pair_analysis['Wobble']:
                    with st.expander("View all base pairs"):
                        if pair_analysis['Watson-Crick']:
                            st.markdown("**Watson-Crick pairs:**")
                            for start, end, pair_type in pair_analysis['Watson-Crick']:
                                st.markdown(f"• {start}-{end}: `{pair_type}`")
                        
                        if pair_analysis['Wobble']:
                            st.markdown("**Wobble pairs:**")
                            for start, end, pair_type in pair_analysis['Wobble']:
                                st.markdown(f"• {start}-{end}: `{pair_type}` (wobble)")
            
            # Structure quality assessment
            st.markdown('<div class="section-header">⭐ Structure Quality Assessment</div>', unsafe_allow_html=True)
            
            # Calculate quality metrics
            quality_score = 0
            quality_factors = []
            
            # Factor 1: Pairing efficiency
            if pairing_efficiency > 60:
                quality_score += 30
                quality_factors.append("✅ High pairing efficiency")
            elif pairing_efficiency > 40:
                quality_score += 20
                quality_factors.append("⚠️ Moderate pairing efficiency")
            else:
                quality_score += 10
                quality_factors.append("❌ Low pairing efficiency")
            
            # Factor 2: Energy stability
            if energy < -15:
                quality_score += 30
                quality_factors.append("✅ Very stable structure")
            elif energy < -8:
                quality_score += 20
                quality_factors.append("⚠️ Moderately stable structure")
            else:
                quality_score += 10
                quality_factors.append("❌ Less stable structure")
            
            # Factor 3: Structural diversity
            if len(hairpins) > 0 and len(internal_loops) > 0:
                quality_score += 25
                quality_factors.append("✅ Diverse structural elements")
            elif len(hairpins) > 0 or len(internal_loops) > 0:
                quality_score += 15
                quality_factors.append("⚠️ Some structural diversity")
            else:
                quality_score += 5
                quality_factors.append("❌ Limited structural diversity")
            
            # Factor 4: GC content balance
            if 40 <= gc_content <= 60:
                quality_score += 15
                quality_factors.append("✅ Balanced GC content")
            elif 30 <= gc_content <= 70:
                quality_score += 10
                quality_factors.append("⚠️ Acceptable GC content")
            else:
                quality_score += 5
                quality_factors.append("❌ Extreme GC content")
            
            # Display quality assessment
            quality_grade = "Excellent" if quality_score >= 85 else "Good" if quality_score >= 70 else "Fair" if quality_score >= 55 else "Poor"
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Quality Score", f"{quality_score}/100")
                st.metric("Grade", quality_grade)
                
                # Quality indicator
                if quality_score >= 85:
                    st.success("🎉 Excellent structure prediction!")
                elif quality_score >= 70:
                    st.info("👍 Good structure prediction")
                elif quality_score >= 55:
                    st.warning("⚠️ Fair structure prediction")
                else:
                    st.error("❌ Poor structure prediction")
            
            with col2:
                st.markdown("**Quality Factors:**")
                for factor in quality_factors:
                    st.markdown(f"• {factor}")
                
                st.markdown("**Recommendations:**")
                if quality_score < 70:
                    st.markdown("• Consider using more sophisticated folding algorithms")
                    st.markdown("• Verify sequence accuracy")
                    if gc_content < 30 or gc_content > 70:
                        st.markdown("• Review GC content - extreme values may affect folding")
            
            # Export options
            st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Detailed text report
                detailed_report = f"""RNA Secondary Structure Analysis Report
{'='*50}

SEQUENCE INFORMATION:
Length: {len(processed_seq)} nucleotides
GC Content: {gc_content:.1f}%
Sequence: {processed_seq}

STRUCTURE PREDICTION:
Dot-bracket notation: {structure}
Estimated free energy (ΔG): {energy:.2f} kcal/mol
Stability: {stability}

STRUCTURAL STATISTICS:
Total base pairs: {base_pairs}
Pairing efficiency: {pairing_efficiency:.1f}%
Maximum nesting depth: {max_depth}
Unpaired bases: {structure.count('.')}

BASE COMPOSITION:
{chr(10).join([f"{base}: {count} ({count/len(processed_seq)*100:.1f}%)" for base, count in composition.items()])}

BASE PAIRING ANALYSIS:
Watson-Crick pairs: {len(pair_analysis['Watson-Crick'])}
Wobble pairs: {len(pair_analysis['Wobble'])}

STRUCTURAL ELEMENTS:
Hairpin loops: {len(hairpins)}
Internal loops/bulges: {len(internal_loops)}

QUALITY ASSESSMENT:
Quality score: {quality_score}/100
Grade: {quality_grade}
Quality factors:
{chr(10).join([f"  - {factor}" for factor in quality_factors])}

HAIRPIN LOOP DETAILS:
{chr(10).join([f"  Hairpin {i+1}: positions {h.start+1}-{h.end+1}, sequence: {h.sequence}, size: {h.loop_size}" for i, h in enumerate(hairpins)])}

INTERNAL LOOP/BULGE DETAILS:
{chr(10).join([f"  Loop {i+1}: positions {l.start+1}-{l.end+1}, sequence: {l.sequence}, size: {l.loop_size}" for i, l in enumerate(internal_loops)])}

Generated by Enhanced RNA Structure Predictor
Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                st.download_button(
                    label="📄 Download Detailed Report",
                    data=detailed_report,
                    file_name=f"rna_structure_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # CSV format for data analysis
                csv_data = []
                csv_data.append(['Metric', 'Value'])
                csv_data.append(['Sequence_Length', len(processed_seq)])
                csv_data.append(['GC_Content_Percent', gc_content])
                csv_data.append(['Base_Pairs', base_pairs])
                csv_data.append(['Free_Energy_kcal_mol', energy])
                csv_data.append(['Pairing_Efficiency_Percent', pairing_efficiency])
                csv_data.append(['Max_Nesting_Depth', max_depth])
                csv_data.append(['Hairpin_Loops', len(hairpins)])
                csv_data.append(['Internal_Loops', len(internal_loops)])
                csv_data.append(['Watson_Crick_Pairs', len(pair_analysis['Watson-Crick'])])
                csv_data.append(['Wobble_Pairs', len(pair_analysis['Wobble'])])
                csv_data.append(['Quality_Score', quality_score])
                
                df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_string,
                    file_name=f"rna_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # FASTA format
                fasta_content = f""">RNA_Structure_Prediction|Length={len(processed_seq)}|GC={gc_content:.1f}%|Energy={energy:.2f}
{processed_seq}
>Structure_Dot_Bracket|BasePairs={base_pairs}|Quality={quality_grade}
{structure}
"""
                
                st.download_button(
                    label="🧬 Download FASTA Format",
                    data=fasta_content,
                    file_name=f"rna_structure_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.fasta",
                    mime="text/plain"
                )
            
        else:
            invalid_chars = predictor.get_invalid_characters(processed_seq)
            if invalid_chars:
                st.error(f"❌ Invalid characters found: {', '.join(invalid_chars)}")
                st.info("💡 Please use only A, U, G, C nucleotides. Enable 'Convert T to U' if you have a DNA sequence.")
            else:
                st.error("❌ Invalid RNA sequence! Please use only A, U, G, C nucleotides.")
                st.info("💡 Tip: Enable 'Convert T to U' if you have a DNA sequence.")
    
    # Information section
    st.markdown('<div class="section-header">ℹ️ About This Tool</div>', unsafe_allow_html=True)
    
    with st.expander("How the prediction algorithm works"):
        st.markdown("""
        **Enhanced Dynamic Programming Algorithm:**
        
        1. **Energy Model**: Uses simplified thermodynamic parameters including:
           - Base pairing energies (GC: -3.0, AU: -2.0, GU: -1.0 kcal/mol)
           - Stacking interactions between adjacent base pairs
           - Loop penalties based on loop size and type
        
        2. **Structure Prediction**: 
           - Nussinov-style dynamic programming with energy optimization
           - Considers hairpin loops, internal loops, and bulges
           - Minimum loop size of 3 nucleotides enforced
        
        3. **Visualization**: 
           - Multiple layout algorithms (hierarchical and radial)
           - Automatic detection and labeling of structural elements
           - Professional-style 2D diagrams with base-specific coloring
        
        **Limitations:**
        - Simplified energy model (not as accurate as ViennaRNA or RNAfold)
        - Does not predict pseudoknots
        - Assumes single optimal structure (no ensemble)
        - Best for educational purposes and preliminary analysis
        """)
    
    with st.expander("Interpreting the results"):
        st.markdown("""
        **Energy Values:**
        - More negative = more stable structure
        - Typical range: -5 to -30 kcal/mol for natural RNAs
        
        **Structural Elements:**
        - **Hairpin loops**: Small loops (3-8 nt) closed by a stem
        - **Internal loops**: Larger unpaired regions within stems
        - **Bulges**: Small asymmetric loops (1-3 nt)
        - **Stems**: Consecutive base-paired regions
        
        **Quality Indicators:**
        - **Pairing efficiency**: Percentage of bases involved in pairs
        - **GC content**: Affects overall stability (optimal: 40-60%)
        - **Structural diversity**: Presence of various element types
        
        **Color Coding:**
        - Red: Adenine (A)
        - Blue: Uracil (U) or Thymine (T)
        - Green: Guanine (G)  
        - Orange: Cytosine (C)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>Enhanced RNA Structure Predictor v2.0</strong></p>
        <p>Features professional 2D visualization, comprehensive analysis, and detailed quality assessment</p>
        <p><strong>Note:</strong> For research applications, please use specialized tools like RNAfold, Mfold, or RNAstructure</p>
        <p>Built with ❤️ using Streamlit and advanced visualization techniques</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
