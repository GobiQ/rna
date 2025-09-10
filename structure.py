import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from io import StringIO
import re
import base64
import math

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

class RNAStructurePredictor:
    """Simple RNA secondary structure predictor using basic thermodynamic principles"""
    
    def __init__(self):
        # Base pairing rules
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
    
    def is_valid_rna(self, sequence):
        """Validate RNA sequence (allows T for DNA sequences)"""
        return bool(re.match(r'^[AUTGC]+$', sequence.upper()))
    
    def can_pair(self, base1, base2):
        """Check if two bases can pair"""
        return (base1, base2) in self.base_pairs
    
    def get_pairing_energy(self, base1, base2):
        """Get energy for base pairing"""
        return self.base_pairs.get((base1, base2), 0)
    
    def simple_fold(self, sequence):
        """
        Simple RNA folding algorithm using dynamic programming (simplified Nussinov)
        Returns dot-bracket notation and estimated free energy
        """
        n = len(sequence)
        if n < 4:
            return '.' * n, 0.0
        
        # DP table for maximum number of base pairs
        dp = [[0 for _ in range(n)] for _ in range(n)]
        traceback = [[None for _ in range(n)] for _ in range(n)]
        
        # Fill DP table
        for length in range(4, n + 1):  # minimum loop size of 3
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Option 1: don't pair i and j
                if j > i:
                    dp[i][j] = dp[i][j-1]
                
                # Option 2: pair i and j (if possible)
                if self.can_pair(sequence[i], sequence[j]) and j - i >= 3:
                    energy_gain = 1  # simplified scoring
                    if dp[i+1][j-1] + energy_gain > dp[i][j]:
                        dp[i][j] = dp[i+1][j-1] + energy_gain
                        traceback[i][j] = 'pair'
                
                # Option 3: split into two subproblems
                for k in range(i, j):
                    if dp[i][k] + dp[k+1][j] > dp[i][j]:
                        dp[i][j] = dp[i][k] + dp[k+1][j]
                        traceback[i][j] = k
        
        # Traceback to get structure
        structure = ['.'] * n
        self._traceback(sequence, traceback, structure, 0, n-1)
        
        # Calculate approximate free energy
        free_energy = self._calculate_energy(sequence, ''.join(structure))
        
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
    
    def _calculate_energy(self, sequence, structure):
        """Calculate approximate free energy"""
        energy = 0.0
        stack = []
        
        for i, symbol in enumerate(structure):
            if symbol == '(':
                stack.append(i)
            elif symbol == ')' and stack:
                j = stack.pop()
                # Base pairing energy
                energy += self.get_pairing_energy(sequence[j], sequence[i])
        
        return energy
    
    def analyze_structure_elements(self, sequence, structure):
        """Analyze structural elements like hairpins, bulges, etc."""
        elements = []
        stack = []
        
        for i, symbol in enumerate(structure):
            if symbol == '(':
                stack.append(i)
            elif symbol == ')' and stack:
                start = stack.pop()
                stem_length = 1
                
                # Check if this is part of a longer stem
                j = start - 1
                k = i + 1
                while (j >= 0 and k < len(structure) and 
                       structure[j] == '(' and structure[k] == ')'):
                    stem_length += 1
                    j -= 1
                    k += 1
                
                # Calculate loop size
                loop_size = i - start - 1
                
                if loop_size <= 6:  # Hairpin loop
                    elements.append({
                        'type': 'hairpin',
                        'start': start,
                        'end': i,
                        'stem_length': stem_length,
                        'loop_size': loop_size
                    })
                elif loop_size > 6:  # Internal loop or bulge
                    elements.append({
                        'type': 'internal_loop',
                        'start': start,
                        'end': i,
                        'stem_length': stem_length,
                        'loop_size': loop_size
                    })
        
        return elements

def analyze_sequence(sequence):
    """Analyze RNA sequence composition"""
    sequence_upper = sequence.upper()
    composition = {base: sequence_upper.count(base) for base in 'AUGC'}
    gc_content = (composition['G'] + composition['C']) / len(sequence) * 100
    
    return composition, gc_content

def create_structure_visualization(sequence, structure, energy):
    """Create comprehensive RNA structure visualization"""
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1], width_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])  # Base composition
    ax2 = fig.add_subplot(gs[0, 1])  # GC content pie chart
    ax3 = fig.add_subplot(gs[1, :])  # Structure arc diagram
    ax4 = fig.add_subplot(gs[2, :])  # Linear structure plot
    
    # Plot 1: Base composition
    bases = ['A', 'U', 'G', 'C']
    counts = [sequence.count(base) for base in bases]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    bars = ax1.bar(bases, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_title('Base Composition', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: GC content
    gc_count = sequence.count('G') + sequence.count('C')
    at_count = len(sequence) - gc_count
    
    wedges, texts, autotexts = ax2.pie([gc_count, at_count], 
                                      labels=['GC', 'AU'], 
                                      colors=['#27ae60', '#e74c3c'],
                                      autopct='%1.1f%%',
                                      startangle=90)
    ax2.set_title('GC Content', fontsize=14, fontweight='bold')
    
    # Plot 3: Arc diagram showing base pairs
    ax3.set_xlim(0, len(sequence))
    ax3.set_ylim(0, len(sequence) // 4)
    
    # Draw sequence
    for i, base in enumerate(sequence):
        color = {'A': '#e74c3c', 'U': '#3498db', 'G': '#2ecc71', 'C': '#f39c12'}[base]
        ax3.scatter(i, 0, c=color, s=80, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.text(i, -0.5, base, ha='center', va='center', fontweight='bold', fontsize=8)
    
    # Draw base pair arcs
    stack = []
    pair_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    color_idx = 0
    
    for i, symbol in enumerate(structure):
        if symbol == '(':
            stack.append(i)
        elif symbol == ')' and stack:
            start = stack.pop()
            
            # Draw arc
            center = (start + i) / 2
            radius = (i - start) / 2
            height = radius * 0.8
            
            # Create arc
            theta = np.linspace(0, np.pi, 50)
            x_arc = center + radius * np.cos(theta)
            y_arc = height * np.sin(theta)
            
            ax3.plot(x_arc, y_arc, color=pair_colors[color_idx % len(pair_colors)], 
                    linewidth=2.5, alpha=0.8)
            
            # Mark base pair with lines
            ax3.plot([start, start], [0, height * 0.1], 'k-', linewidth=1.5, alpha=0.6)
            ax3.plot([i, i], [0, height * 0.1], 'k-', linewidth=1.5, alpha=0.6)
            
            color_idx += 1
    
    ax3.set_title('RNA Secondary Structure - Arc Diagram', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sequence Position')
    ax3.set_ylabel('Base Pair Distance')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1, len(sequence))
    
    # Plot 4: Linear structure representation
    x = range(len(sequence))
    y = []
    depth = 0
    
    for char in structure:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        y.append(depth)
    
    # Create gradient fill
    ax4.plot(x, y, 'b-', linewidth=3, alpha=0.8)
    ax4.fill_between(x, y, alpha=0.4, color='lightblue')
    
    # Highlight different structural regions
    current_depth = 0
    for i, (pos, depth_val) in enumerate(zip(x, y)):
        if depth_val != current_depth:
            if depth_val > current_depth:  # Opening
                ax4.axvline(pos, color='green', alpha=0.6, linestyle='--', linewidth=1)
            else:  # Closing
                ax4.axvline(pos, color='red', alpha=0.6, linestyle='--', linewidth=1)
            current_depth = depth_val
    
    ax4.set_title(f'Structure Depth Profile (ΔG ≈ {energy:.2f} kcal/mol)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sequence Position')
    ax4.set_ylabel('Nesting Depth')
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for structural features
    base_pairs = structure.count('(')
    max_depth = max(y) if y else 0
    
    textstr = f'Base Pairs: {base_pairs}\nMax Nesting: {max_depth}\nLength: {len(sequence)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def create_detailed_structure_analysis(sequence, structure):
    """Create detailed analysis of structural elements"""
    predictor = RNAStructurePredictor()
    elements = predictor.analyze_structure_elements(sequence, structure)
    
    analysis = {
        'hairpins': [],
        'internal_loops': [],
        'stems': []
    }
    
    # Analyze each element
    for element in elements:
        if element['type'] == 'hairpin':
            loop_seq = sequence[element['start']+1:element['end']]
            analysis['hairpins'].append({
                'position': f"{element['start']}-{element['end']}",
                'loop_sequence': loop_seq,
                'loop_size': element['loop_size'],
                'stem_length': element['stem_length']
            })
        elif element['type'] == 'internal_loop':
            loop_seq = sequence[element['start']+1:element['end']]
            analysis['internal_loops'].append({
                'position': f"{element['start']}-{element['end']}",
                'loop_sequence': loop_seq,
                'loop_size': element['loop_size']
            })
    
    # Find stems
    stack = []
    for i, symbol in enumerate(structure):
        if symbol == '(':
            stack.append(i)
        elif symbol == ')' and stack:
            start = stack.pop()
            base_pair = f"{sequence[start]}-{sequence[i]}"
            analysis['stems'].append({
                'position': f"{start}-{i}",
                'base_pair': base_pair,
                'pair_type': 'Watson-Crick' if (sequence[start], sequence[i]) in [('G','C'), ('C','G'), ('A','U'), ('U','A')] else 'Wobble'
            })
    
    return analysis

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 RNA Structure Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts RNA secondary structure using a simplified dynamic programming algorithm 
    based on thermodynamic principles. Enter an RNA sequence to get structure predictions and detailed analysis.
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Options")
        
        # Example sequences
        st.markdown("#### Example Sequences")
        examples = {
            "tRNA-like": "GCGCAAUUAGGCGCGUCCCUCCACCCUGUGCCUUUCCAGGGCUGGGCAAGAUUCUGCGAACGACUCCCGCCGUGUUU",
            "Hairpin": "GGCUUUAGCCUUCGCCACCAUGAGCGUGGUACCUCCGAGCUUCGAGCGGCCCC",
            "Simple Hairpin": "GGGGAAAACCCC",
            "Complex Structure": "CUGCUGUCAGCCGGACUACUUGUGCGCGCAAUAACCCUAGGGCUGCCUUCGGGAACCUGUCUGUAA",
            "Multiple Hairpins": "GGCUUUAGCCCCAAAAGGGGUUUUCCCC"
        }
        
        selected_example = st.selectbox("Choose an example:", ["None"] + list(examples.keys()))
        
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
            st.markdown(f'<div class="sequence-display">{processed_seq}</div>', unsafe_allow_html=True)
            
            # Sequence analysis
            composition, gc_content = analyze_sequence(processed_seq)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", len(processed_seq))
            with col2:
                st.metric("GC Content", f"{gc_content:.1f}%")
            with col3:
                st.metric("Base Pairs (predicted)", "Calculating...")
            
            # Structure prediction
            with st.spinner("Predicting RNA structure..."):
                structure, energy = predictor.simple_fold(processed_seq)
            
            # Update base pairs metric
            base_pairs = structure.count('(')
            col3.metric("Base Pairs (predicted)", base_pairs)
            
            # Display results
            st.markdown('<div class="section-header">Structure Prediction Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sequence:**")
                st.markdown(f'<div class="sequence-display">{processed_seq}</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown("**Structure (Dot-Bracket):**")
                st.markdown(f'<div class="structure-display">{structure}</div>', unsafe_allow_html=True)
            
            # Energy information
            st.markdown(f'''
            <div class="energy-box">
                <h4>Thermodynamic Information</h4>
                <p><strong>Estimated Free Energy (ΔG):</strong> {energy:.2f} kcal/mol</p>
                <p><strong>Base Pairs:</strong> {base_pairs}</p>
                <p><strong>Pairing Efficiency:</strong> {(base_pairs * 2 / len(processed_seq) * 100):.1f}%</p>
                <p><strong>Structure Stability:</strong> {'High' if energy < -10 else 'Medium' if energy < -5 else 'Low'}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Comprehensive Visualization
            st.markdown('<div class="section-header">Comprehensive Structure Visualization</div>', unsafe_allow_html=True)
            
            fig = create_structure_visualization(processed_seq, structure, energy)
            st.pyplot(fig)
            
            # Detailed structural analysis
            st.markdown('<div class="section-header">Structural Elements Analysis</div>', unsafe_allow_html=True)
            
            analysis = create_detailed_structure_analysis(processed_seq, structure)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🎀 Hairpin Loops")
                if analysis['hairpins']:
                    for i, hairpin in enumerate(analysis['hairpins']):
                        st.markdown(f"**Hairpin {i+1}:**")
                        st.markdown(f"- Position: {hairpin['position']}")
                        st.markdown(f"- Loop sequence: `{hairpin['loop_sequence']}`")
                        st.markdown(f"- Loop size: {hairpin['loop_size']} nt")
                        st.markdown(f"- Stem length: {hairpin['stem_length']} bp")
                        st.markdown("---")
                else:
                    st.markdown("No hairpin loops detected")
            
            with col2:
                st.markdown("#### 🔄 Internal Loops")
                if analysis['internal_loops']:
                    for i, loop in enumerate(analysis['internal_loops']):
                        st.markdown(f"**Internal Loop {i+1}:**")
                        st.markdown(f"- Position: {loop['position']}")
                        st.markdown(f"- Loop sequence: `{loop['loop_sequence']}`")
                        st.markdown(f"- Size: {loop['loop_size']} nt")
                        st.markdown("---")
                else:
                    st.markdown("No internal loops detected")
            
            with col3:
                st.markdown("#### 🔗 Base Pairs")
                if analysis['stems']:
                    pair_counts = {'Watson-Crick': 0, 'Wobble': 0}
                    for stem in analysis['stems']:
                        pair_counts[stem['pair_type']] += 1
                    
                    st.markdown(f"**Total pairs:** {len(analysis['stems'])}")
                    st.markdown(f"**Watson-Crick:** {pair_counts['Watson-Crick']}")
                    st.markdown(f"**Wobble pairs:** {pair_counts['Wobble']}")
                    
                    with st.expander("View all base pairs"):
                        for stem in analysis['stems']:
                            st.markdown(f"- {stem['position']}: `{stem['base_pair']}` ({stem['pair_type']})")
                else:
                    st.markdown("No base pairs detected")
            
            # Detailed analysis
            with st.expander("📊 Complete Sequence Analysis"):
                st.markdown("#### Base Composition")
                comp_df = pd.DataFrame([composition]).T
                comp_df.columns = ['Count']
                comp_df['Percentage'] = comp_df['Count'] / len(processed_seq) * 100
                st.dataframe(comp_df)
                
                st.markdown("#### Structure Statistics")
                struct_stats = {
                    'Unpaired bases': structure.count('.'),
                    'Opening brackets': structure.count('('),
                    'Closing brackets': structure.count(')'),
                    'Total base pairs': base_pairs,
                    'Maximum nesting depth': max([structure[:i+1].count('(') - structure[:i+1].count(')') for i in range(len(structure))]),
                    'Average loop size': np.mean([len(loop['loop_sequence']) for loop in analysis['hairpins'] + analysis['internal_loops']]) if (analysis['hairpins'] or analysis['internal_loops']) else 0
                }
                st.json(struct_stats)
                
            # Download results
            st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
            
            results_text = f"""RNA Structure Prediction Results
================================
Sequence: {processed_seq}
Structure: {structure}
Length: {len(processed_seq)}
Base pairs: {base_pairs}
GC content: {gc_content:.1f}%
Estimated ΔG: {energy:.2f} kcal/mol

Base composition:
{chr(10).join([f"{base}: {count} ({count/len(processed_seq)*100:.1f}%)" for base, count in composition.items()])}

Structural Elements:
Hairpins: {len(analysis['hairpins'])}
Internal loops: {len(analysis['internal_loops'])}
Total base pairs: {len(analysis['stems'])}
"""
            
            st.download_button(
                label="📥 Download Results (TXT)",
                data=results_text,
                file_name="rna_structure_results.txt",
                mime="text/plain"
            )
            
        else:
            st.error("❌ Invalid RNA sequence! Please use only A, U, G, C nucleotides.")
            st.info("💡 Tip: Enable 'Convert T to U' if you have a DNA sequence.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>Note:</strong> This is a simplified RNA structure predictor for educational purposes. 
        For research applications, please use specialized tools like RNAfold, Mfold, or RNAstructure.</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
