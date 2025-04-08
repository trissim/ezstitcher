"""
CSV handler for ezstitcher.

This module provides a class for handling CSV file operations.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CSVHandler:
    """Handle CSV file operations for position files."""
    
    @staticmethod
    def parse_positions_csv(csv_path):
        """
        Parse a CSV file with lines of the form:
          file: <filename>; grid: (col, row); position: (x, y)
        
        Args:
            csv_path (str or Path): Path to the CSV file
            
        Returns:
            list: List of tuples (filename, x_float, y_float)
        """
        entries = []
        with open(csv_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # Example line:
                # file: some_image.tif; grid: (0, 0); position: (123.45, 67.89)
                file_match = re.search(r'file:\s*([^;]+);', line)
                pos_match = re.search(r'position:\s*\(([^,]+),\s*([^)]+)\)', line)
                if file_match and pos_match:
                    fname = file_match.group(1).strip()
                    x_val = float(pos_match.group(1).strip())
                    y_val = float(pos_match.group(2).strip())
                    entries.append((fname, x_val, y_val))
        return entries
    
    @staticmethod
    def generate_positions_df(image_files, positions, grid_positions):
        """
        Generate a DataFrame with position information.
        
        Args:
            image_files (list): List of image filenames
            positions (list): List of (x, y) position tuples
            grid_positions (list): List of (row, col) grid position tuples
            
        Returns:
            pandas.DataFrame: DataFrame with position information
        """
        import pandas as pd
        
        # Ensure we don't try to access beyond the available positions
        num_positions = min(len(image_files), len(positions), len(grid_positions))
        data_rows = []
        
        for i in range(num_positions):
            fname = image_files[i]
            x, y = positions[i]
            row, col = grid_positions[i]
            
            data_rows.append({
                "file": "file: " + fname,
                "grid": " grid: " + "("+str(row)+", "+str(col)+")",
                "position": " position: " + "("+str(x)+", "+str(y)+")",
            })
        
        df = pd.DataFrame(data_rows)
        return df
    
    @staticmethod
    def save_positions_df(df, positions_path):
        """
        Save a positions DataFrame to CSV.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            positions_path (str or Path): Path to save the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            Path(positions_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(positions_path, index=False, sep=";", header=False)
            return True
        except Exception as e:
            logger.error(f"Error saving positions CSV: {e}")
            return False
