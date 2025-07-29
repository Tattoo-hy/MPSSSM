#!/usr/bin/env python3
"""
MPS-SSM Results Summarization Script
Aggregates experimental results for multiple dataset types
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse


class ResultsSummarizer:
    """Summarize and format experimental results across multiple datasets"""
    
    def __init__(self, results_dir: str = "results/final_runs/logs", include_all: bool = False):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.include_all = include_all
        
    def load_results(self):
        """Load all result JSON files"""
        for result_file in self.results_dir.glob("*.json"):
            with open(result_file, 'r') as f:
                result = json.load(f)
                self.results_data.append(result)
                
        print(f"Loaded {len(self.results_data)} result files")
        
    def get_dataset_type(self, dataset_name: str) -> str:
        """Determine dataset type from name"""
        if dataset_name.startswith('ETT'):
            return 'ETT'
        elif dataset_name == 'weather':
            return 'Weather'
        elif dataset_name == 'traffic':
            return 'Traffic'
        else:
            return 'Unknown'
            
    def create_main_table(self) -> pd.DataFrame:
        """Create main results table"""
        rows = []
        
        for result in self.results_data:
            row = {
                'Dataset': result['dataset'],
                'Type': self.get_dataset_type(result['dataset']),
                'Pred_Len': result['pred_len'],
                'Lambda': result['best_lambda'],
                'Test_MSE': f"{result['test_mse']:.4f}",
                'Test_MAE': f"{result['test_mae']:.4f}",
                'Impulse_MSE': f"{result['robustness']['impulse_mse']:.4f}",
                'Impulse_Degrad': f"{result['robustness']['impulse_degradation']:.2%}",
                'Spurious_MSE': f"{result['robustness']['spurious_mse']:.4f}",
                'Spurious_Degrad': f"{result['robustness']['spurious_degradation']:.2%}"
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        # Sort by dataset type and prediction length
        df = df.sort_values(['Type', 'Dataset', 'Pred_Len'])
        
        return df
        
    def create_comparison_table(self) -> pd.DataFrame:
        """Create averaged comparison table across datasets"""
        rows = []
        
        # Group by dataset
        dataset_groups = {}
        for result in self.results_data:
            dataset = result['dataset']
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(result)
            
        # Calculate averages for each dataset
        for dataset, results in dataset_groups.items():
            avg_mse = sum(r['test_mse'] for r in results) / len(results)
            avg_mae = sum(r['test_mae'] for r in results) / len(results)
            avg_impulse_deg = sum(r['robustness']['impulse_degradation'] for r in results) / len(results)
            avg_spurious_deg = sum(r['robustness']['spurious_degradation'] for r in results) / len(results)
            
            row = {
                'Dataset': dataset,
                'Type': self.get_dataset_type(dataset),
                'Avg_MSE': f"{avg_mse:.4f}",
                'Avg_MAE': f"{avg_mae:.4f}",
                'Avg_Impulse_Degrad': f"{avg_impulse_deg:.2%}",
                'Avg_Spurious_Degrad': f"{avg_spurious_deg:.2%}",
                'Avg_Total_Degrad': f"{(avg_impulse_deg + avg_spurious_deg) / 2:.2%}"
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df = df.sort_values(['Type', 'Dataset'])
        
        return df
        
    def create_dataset_type_summary(self) -> pd.DataFrame:
        """Create summary by dataset type"""
        rows = []
        
        # Group by dataset type
        type_groups = {'ETT': [], 'Weather': [], 'Traffic': []}
        for result in self.results_data:
            dtype = self.get_dataset_type(result['dataset'])
            if dtype in type_groups:
                type_groups[dtype].append(result)
                
        # Calculate averages for each type
        for dtype, results in type_groups.items():
            if results:
                avg_mse = sum(r['test_mse'] for r in results) / len(results)
                avg_mae = sum(r['test_mae'] for r in results) / len(results)
                avg_impulse_deg = sum(r['robustness']['impulse_degradation'] for r in results) / len(results)
                avg_spurious_deg = sum(r['robustness']['spurious_degradation'] for r in results) / len(results)
                
                row = {
                    'Dataset_Type': dtype,
                    'Num_Experiments': len(results),
                    'Avg_MSE': f"{avg_mse:.4f}",
                    'Avg_MAE': f"{avg_mae:.4f}",
                    'Avg_Impulse_Degrad': f"{avg_impulse_deg:.2%}",
                    'Avg_Spurious_Degrad': f"{avg_spurious_deg:.2%}",
                    'Avg_Robustness': f"{(avg_impulse_deg + avg_spurious_deg) / 2:.2%}"
                }
                rows.append(row)
                
        df = pd.DataFrame(rows)
        df = df.sort_values('Dataset_Type')
        
        return df
        
    def create_lambda_analysis(self) -> pd.DataFrame:
        """Analyze optimal lambda values"""
        rows = []
        
        for result in self.results_data:
            row = {
                'Dataset': result['dataset'],
                'Type': self.get_dataset_type(result['dataset']),
                'Pred_Len': result['pred_len'],
                'Optimal_Lambda': result['best_lambda']
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df_pivot = df.pivot(index='Dataset', columns='Pred_Len', values='Optimal_Lambda')
        
        return df_pivot
        
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report"""
        report = []
        
        # Header
        report.append("# MPS-SSM Multivariate Prediction Results Summary\n")
        report.append("## Overview\n")
        report.append(f"Total experiments analyzed: {len(self.results_data)}\n")
        
        # Dataset type counts
        type_counts = {}
        for result in self.results_data:
            dtype = self.get_dataset_type(result['dataset'])
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        report.append("### Experiments by Dataset Type\n")
        for dtype, count in sorted(type_counts.items()):
            report.append(f"- {dtype}: {count} experiments\n")
        report.append("\n")
        
        # Dataset type summary
        report.append("## Performance Summary by Dataset Type\n")
        type_summary = self.create_dataset_type_summary()
        report.append(type_summary.to_markdown(index=False))
        report.append("\n")
        
        # Main results table
        report.append("## Detailed Results by Dataset and Prediction Length\n")
        main_table = self.create_main_table()
        report.append(main_table.to_markdown(index=False))
        report.append("\n")
        
        # Comparison table
        report.append("## Average Performance Across Prediction Lengths\n")
        comparison_table = self.create_comparison_table()
        report.append(comparison_table.to_markdown(index=False))
        report.append("\n")
        
        # Lambda analysis
        report.append("## Optimal Lambda Values\n")
        lambda_table = self.create_lambda_analysis()
        report.append(lambda_table.to_markdown())
        report.append("\n")
        
        # Key findings
        report.append("## Key Findings\n")
        report.append(self._generate_key_findings())
        
        return "\n".join(report)
        
    def _generate_key_findings(self) -> str:
        """Generate key findings from results"""
        findings = []
        
        # Calculate overall statistics
        all_impulse_deg = [r['robustness']['impulse_degradation'] for r in self.results_data]
        all_spurious_deg = [r['robustness']['spurious_degradation'] for r in self.results_data]
        
        avg_impulse = sum(all_impulse_deg) / len(all_impulse_deg)
        avg_spurious = sum(all_spurious_deg) / len(all_spurious_deg)
        
        findings.append(f"1. **Overall Robustness Performance**:")
        findings.append(f"   - Average impulse noise degradation: {avg_impulse:.2%}")
        findings.append(f"   - Average spurious correlation degradation: {avg_spurious:.2%}")
        findings.append(f"   - Overall robustness score: {(avg_impulse + avg_spurious) / 2:.2%}\n")
        
        # Performance by dataset type
        type_performance = {}
        for result in self.results_data:
            dtype = self.get_dataset_type(result['dataset'])
            if dtype not in type_performance:
                type_performance[dtype] = {'impulse': [], 'spurious': []}
            type_performance[dtype]['impulse'].append(result['robustness']['impulse_degradation'])
            type_performance[dtype]['spurious'].append(result['robustness']['spurious_degradation'])
            
        findings.append(f"2. **Performance by Dataset Type**:")
        for dtype, perfs in sorted(type_performance.items()):
            if perfs['impulse']:
                avg_imp = sum(perfs['impulse']) / len(perfs['impulse'])
                avg_spu = sum(perfs['spurious']) / len(perfs['spurious'])
                findings.append(f"   - {dtype}: {(avg_imp + avg_spu) / 2:.2%} average degradation")
        findings.append("")
        
        # Best and worst performing datasets
        dataset_performance = {}
        for result in self.results_data:
            dataset = result['dataset']
            if dataset not in dataset_performance:
                dataset_performance[dataset] = []
            total_deg = (result['robustness']['impulse_degradation'] + 
                        result['robustness']['spurious_degradation']) / 2
            dataset_performance[dataset].append(total_deg)
            
        avg_performance = {d: sum(perfs)/len(perfs) 
                          for d, perfs in dataset_performance.items()}
        
        best_dataset = min(avg_performance, key=avg_performance.get)
        worst_dataset = max(avg_performance, key=avg_performance.get)
        
        findings.append(f"3. **Best and Worst Performers**:")
        findings.append(f"   - Most robust: {best_dataset} ({avg_performance[best_dataset]:.2%} degradation)")
        findings.append(f"   - Least robust: {worst_dataset} ({avg_performance[worst_dataset]:.2%} degradation)\n")
        
        # Lambda analysis
        all_lambdas = [r['best_lambda'] for r in self.results_data]
        findings.append(f"4. **Hyperparameter Insights**:")
        findings.append(f"   - Lambda range: [{min(all_lambdas)}, {max(all_lambdas)}]")
        findings.append(f"   - Most common lambda: {max(set(all_lambdas), key=all_lambdas.count)}")
        
        # Dataset-specific insights
        findings.append(f"\n5. **Dataset-Specific Insights**:")
        for dtype in ['ETT', 'Weather', 'Traffic']:
            dtype_results = [r for r in self.results_data if self.get_dataset_type(r['dataset']) == dtype]
            if dtype_results:
                avg_mse = sum(r['test_mse'] for r in dtype_results) / len(dtype_results)
                findings.append(f"   - {dtype}: Average MSE = {avg_mse:.4f}")
        
        return "\n".join(findings)
        
    def save_results(self, output_dir: str = "results"):
        """Save all results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save markdown report
        report = self.generate_markdown_report()
        with open(output_path / "results_summary.md", 'w') as f:
            f.write(report)
            
        # Save CSV tables
        self.create_main_table().to_csv(output_path / "detailed_results.csv", index=False)
        self.create_comparison_table().to_csv(output_path / "comparison_results.csv", index=False)
        self.create_lambda_analysis().to_csv(output_path / "lambda_analysis.csv")
        self.create_dataset_type_summary().to_csv(output_path / "dataset_type_summary.csv", index=False)
        
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize MPS-SSM experimental results")
    parser.add_argument('--results_dir', type=str, 
                       default="results/final_runs/logs",
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, 
                       default="results",
                       help='Directory to save summary files')
    parser.add_argument('--include_all_datasets', action='store_true',
                       help='Include all dataset types in summary')
    
    args = parser.parse_args()
    
    summarizer = ResultsSummarizer(args.results_dir, args.include_all_datasets)
    summarizer.load_results()
    summarizer.save_results(args.output_dir)
    
    # Print summary to console
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(summarizer.generate_markdown_report())


if __name__ == "__main__":
    main()