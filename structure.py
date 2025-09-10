import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re
import base64

# Set page config
st.set_page_config(
    page_title="RNA Structure Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .sequence-display {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .structure-display {
        font-family: 'Courier New', monospace;
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .energy-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
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

def analyze_sequence(sequence):
    """Analyze RNA sequence composition"""
    composition = {base: sequence.count(base) for base in 'AUGC'}
    gc_content = (composition['G'] + composition['C']) / len(sequence) * 100
    
    return composition, gc_content

def create_structure_plot(sequence, structure, energy):
    """Create a simple visualization of the RNA structure"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Base composition
    bases = ['A', 'U', 'G', 'C']
    counts = [sequence.count(base) for base in bases]
    colors = ['red', 'blue', 'green', 'orange']
    
    ax1.bar(bases, counts, color=colors, alpha=0.7)
    ax1.set_title('Base Composition')
    ax1.set_ylabel('Count')
    
    # Plot 2: Structure representation (simplified)
    x = range(len(sequence))
    y = []
    depth = 0
    
    for char in structure:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        y.append(depth)
    
    ax2.plot(x, y, 'b-', linewidth=2)
    ax2.fill_between(x, y, alpha=0.3)
    ax2.set_title(f'Secondary Structure Profile (ΔG ≈ {energy:.2f} kcal/mol)')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Structure Depth')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 RNA Structure Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts RNA secondary structure using a simplified dynamic programming algorithm 
    based on thermodynamic principles. Enter an RNA sequence to get structure predictions and analysis.
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Options")
        
        # Example sequences
        st.markdown("#### Example Sequences")
        examples = {
            "tRNA-like": "GCGCAAUUAGGCGCGUCCCUCCACCCUGUGCCUUUCCAGGGCUGGGCAAGAUUCUGCGAACGACUCCCGCCGUGUUU",
            "Hairpin": "GGCUUUAGCCUUCGCCACCAUGAGCGUGGUACCUCCGAGCUUCGAGCGGCCCC",
            "Simple": "GGGGAAAACCCC",
            "Complex": "CUGCUGUCAGCCGGACUACUUGUGCGCGCAAUAACCCUAGGGCUGCCUUCGGGAACCUGUCUGUAA"
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
            </div>
            ''', unsafe_allow_html=True)
            
            # Visualization
            st.markdown('<div class="section-header">Structure Visualization</div>', unsafe_allow_html=True)
            
            fig = create_structure_plot(processed_seq, structure, energy)
            st.pyplot(fig)
            
            # Detailed analysis
            with st.expander("📊 Detailed Analysis"):
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
                    'Structure complexity': len(set(structure))
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
