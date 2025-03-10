import argparse
from modules.inference_weather import FCN2_weather
from modules.FCNV2_to_FCN_precip import FCNV1_precip

def main(input_data, output_folder, action, fore_hr, device='cpu'):
    if action=='FCNV2':
        FCN2_weather(input_data, output_folder, fore_hr, device=device)
    elif action=='FCNV1_precip':
        FCNV1_precip(input_data, output_folder, device=device)
    else:
        print('action only FCNV2 or FCNV1_precip')

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True, help="input_data_file (.nc) or input_folder for FCNV1_precip")
    parser.add_argument("--output_folder", required=True, help="output_folder (folder)")
    parser.add_argument("--action", help="FCNV2 or FCNV1_precip", default='FCNV2')
    parser.add_argument("--fore_hr", help="how many hours", default=240)
    parser.add_argument("--device", help="cpu or cuda", default='cpu')
    args = parser.parse_args()
    main(args.input_data, args.output_folder, args.action, args.fore_hr, device=args.device) 