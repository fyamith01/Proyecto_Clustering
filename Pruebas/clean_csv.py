import csv
import re

def clean_csv_file(input_file, output_file):
    """
    Clean the CSV file by removing unnecessary spaces and adding proper headers
    """
    # Define the headers as requested
    headers = [
        "DocEmp", "TipoDocEmp", "NomEmp", "CodclaseEconomicaEmp", "CodSeccionEconomicaEmp", 
        "NitSinDV", "Femenino", "Masculino", "Desconocido", "TarifaA", "TarifaB", "TarifaC", 
        "Jovenes", "AdultoJoven", "Adulto", "AdultoMayor", "TotalAfiliados", "AportePC", 
        "SubsidioPC", "UsosCultura", "UsosEducacion", "UsosSalud", "UsosEmpleo", 
        "PorcentajeFemenino", "PorcentajeMasculino", "PorcentajeDesconocido", 
        "PorcentajeTarifaA", "PorcentajeTarifaB", "PorcentajeTarifaC", "PorcentajeJovenes", 
        "PorcentajeAdultoJoven", "PorcentajeAdulto", "PorcentajeAdultoMayor"
    ]
    
    with open(input_file, 'r', encoding='utf-8-sig') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        # Write headers
        writer = csv.writer(outfile, delimiter=';')
        writer.writerow(headers)
        
        # Process each line
        for line_num, line in enumerate(infile, 1):
            if line_num % 1000 == 0:
                print(f"Processing line {line_num}...")
            
            # Remove BOM and strip whitespace
            line = line.strip()
            if not line:
                continue
            
            # Remove BOM if present (for the first line)
            if line_num == 1 and line.startswith('\ufeff'):
                line = line[1:]
            
            # Remove quotes from the beginning and end of the line
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Split by semicolon
            fields = line.split(';')
            
            # Clean each field
            cleaned_fields = []
            for field in fields:
                # Remove quotes from field if present
                if field.startswith('"') and field.endswith('"'):
                    field = field[1:-1]
                
                # Remove excessive spaces (replace multiple spaces with single space, then strip)
                field = re.sub(r'\s+', ' ', field).strip()
                
                # If field is empty after cleaning, keep it as empty string
                if not field:
                    field = ""
                
                cleaned_fields.append(field)
            
            # Write the cleaned row
            writer.writerow(cleaned_fields)
    
    print(f"CSV file cleaned successfully. Output saved to: {output_file}")

if __name__ == "__main__":
    input_file = r"afiliados_mas_usos.csv"
    output_file = r"datos_clustering.csv"
    
    print("Starting CSV cleaning process...")
    clean_csv_file(input_file, output_file)
    print("Process completed!")