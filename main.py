import streamlit as st
import yaml
from pathlib import Path
from utils.pdf_processing import extract_text_with_ocr
import time

# Import all processors
from processors.material_filling import analyze as analyze_material_filling
from processors.material_name import analyze as analyze_material_name
from processors.pull_to_seat import analyze as analyze_pull_to_seat
from processors.working_temperature import analyze as analyze_working_temperature
from processors.colour import analyze as analyze_colour
from processors.contact_systems import analyze as analyze_contact_systems
from processors.gender import analyze as analyze_gender
from processors.height_mm import analyze as analyze_height_mm
from processors.housing_seal import analyze as analyze_housing_seal
from processors.hv_qualified import analyze as analyze_hv_qualified
from processors.length_mm import analyze as analyze_length_mm
from processors.mechanical_coding import analyze as analyze_mechanical_coding
from processors.number_of_cavities import analyze as analyze_number_of_cavities
from processors.number_of_rows import analyze as analyze_number_of_rows
from processors.pre_assembled import analyze as analyze_pre_assembled
from processors.sealing import analyze as analyze_sealing
from processors.sealing_class import analyze as analyze_sealing_class
from processors.terminal_position_assurance import analyze as analyze_terminal_position_assurance
from processors.connector_type import analyze as analyze_connector_type
from processors.width_mm import analyze as analyze_width_mm
from processors.wire_seal import analyze as analyze_wire_seal
from processors.connector_position_assurance import analyze as analyze_connector_position_assurance
from processors.colour_coding import analyze as analyze_colour_coding
from processors.set_kit import analyze as analyze_set_kit
from processors.closed_cavities import analyze as analyze_closed_cavities

# Configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
MAX_FILE_SIZE_MB = 50
ALLOWED_TYPES = ["application/pdf"]

def main():
    st.set_page_config(
        page_title="Material Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header Section
    st.title("📄 Combined Document Analyzer")
    st.markdown("Upload multiple technical documents for combined analysis")
    
    # File Upload
    uploaded_files = st.file_uploader(
        "Drag & Drop PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max file size: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_files:
        # Validate files
        valid_files = [
            f for f in uploaded_files 
            if f.type in ALLOWED_TYPES and f.size <= MAX_FILE_SIZE_MB * 1024 * 1024
        ]
        
        with st.status("🔍 Processing documents...", expanded=True) as status:
            combined_text = ""
            processing_times = []
            
            try:
                # First pass: Combine all text
                for file in valid_files:
                    st.write(f"📥 Extracting text from {file.name}...")
                    start_time = time.time()
                    
                    # Extract and combine text
                    file_content = file.read()
                    text = extract_text_with_ocr(file_content)
                    combined_text += f"\n----- {file.name} -----\n{text}"
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    st.success(f"✅ {file.name} processed in {processing_time:.1f}s")
                
                # Second pass: Analyze combined text
                st.write("🔍 Analyzing combined content...")
                analysis_start = time.time()
                
                # Run all analyses
                results = {
                    'material': analyze_material_filling(combined_text),
                    'material_name': analyze_material_name(combined_text),
                    'pull_seat': analyze_pull_to_seat(combined_text),
                    'working_temp': analyze_working_temperature(combined_text),
                    'colour': analyze_colour(combined_text),
                    'contact_systems': analyze_contact_systems(combined_text),
                    'gender': analyze_gender(combined_text),
                    'height': analyze_height_mm(combined_text),
                    'housing_seal': analyze_housing_seal(combined_text),
                    'hv_qualified': analyze_hv_qualified(combined_text),
                    'length': analyze_length_mm(combined_text),
                    'mechanical_coding': analyze_mechanical_coding(combined_text),
                    'cavities': analyze_number_of_cavities(combined_text),
                    'rows': analyze_number_of_rows(combined_text),
                    'pre_assembled': analyze_pre_assembled(combined_text),
                    'sealing': analyze_sealing(combined_text),
                    'sealing_class': analyze_sealing_class(combined_text),
                    'tpa': analyze_terminal_position_assurance(combined_text),
                    'connector_type': analyze_connector_type(combined_text),
                    'width': analyze_width_mm(combined_text),
                    'wire_seal': analyze_wire_seal(combined_text),
                    'cpa': analyze_connector_position_assurance(combined_text),
                    'colour_coding': analyze_colour_coding(combined_text),
                    'set_kit': analyze_set_kit(combined_text),
                    'closed_cavities': analyze_closed_cavities(combined_text)
                }
                
                total_analysis_time = time.time() - analysis_start
                processing_times.append(total_analysis_time)
                
                status.update(label=f"Processed {len(valid_files)} files", state="complete")
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                return
            
        # Display Results
        st.subheader("📊 Combined Analysis Results")
        
        # Summary Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Files", len(valid_files))
        col2.metric("Total Processing Time", f"{sum(processing_times):.1f}s")
        col3.metric("Analysis Time", f"{total_analysis_time:.1f}s")
        
        # Organized Results Display
        with st.container():
            st.markdown("### Material Properties")
            cols = st.columns(4)

            with cols[0]:
                c1, c2 = st.columns([2, 1])
                c1.markdown(f"**Material Filling**\n`{results['material']}`")
                confidence = results.get('material_confidence', 0)
                c2.markdown(f"<br><span style='color:{"#2ecc71" if confidence > 70 else "#e67e22"}'>\n{confidence}%</span>", 
                            unsafe_allow_html=True)

            cols[2].markdown(f"**Colour**\n`{results['colour']}`")
            cols[3].markdown(f"**HV Qualified**\n`{results['hv_qualified']}`")

            st.markdown("### Mechanical Properties")
            cols = st.columns(4)
            cols[0].markdown(f"**Pull-to-Seat**\n`{results['pull_seat']}`")
            cols[1].markdown(f"**Gender**\n`{results['gender']}`")
            cols[2].markdown(f"**Mechanical Coding**\n`{results['mechanical_coding']}`")
            cols[3].markdown(f"**Pre-assembled**\n`{results['pre_assembled']}`")

            st.markdown("### Dimensions")
            cols = st.columns(4)
            cols[0].markdown(f"**Height (mm)**\n`{results['height']}`")
            cols[1].markdown(f"**Width (mm)**\n`{results['width']}`")
            cols[2].markdown(f"**Length (mm)**\n`{results['length']}`")
            cols[3].markdown(f"**Cavities**\n`{results['cavities']}`")

        # Source Documents
        with st.expander("📚 Processed Documents"):
            for file in valid_files:
                st.caption(f"📄 {file.name}")
        
        # Export Options
        csv_data = convert_to_csv(results, [f.name for f in valid_files])
        
        st.download_button(
            label="📥 Export Results as CSV",
            data=csv_data,
            file_name="combined_analysis.csv",
            mime="text/csv"
        )

def convert_to_csv(results, processed_files):
    """Convert results to CSV format"""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["Attribute", "Value"])
    for key, value in results.items():
        writer.writerow([key.replace('_', ' ').title(), value])
    
    writer.writerow([])
    writer.writerow(["Processed Files"] + processed_files)
    
    return output.getvalue()

if __name__ == "__main__":
    main()
