import torch
import argparse
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,help='pretrain model .pt file', default='Unet_best_model')
    parser.add_argument("--input_nc", help="set input_channel", type=int, default=3)
    parser.add_argument("--crop_size", help="set_model_img_size", type=int, default=512)
    args = parser.parse_args()

    if os.path.exists('.\\onnx'):
        print("Output directory already exists: ./onnx/")
    else :
        os.makedirs('.\\onnx')
        print('make Output directory : ./onnx/')

    device = 'cpu'
    model = torch.load('./model/' + args.name + '.pt')
    model.eval()
    model.to(device)

    #batch=1
    dummy_input1 = torch.randn(1, args.input_nc, args.crop_size, args.crop_size, requires_grad=True).to(device)
    dummy_input2 = torch.randn(1, args.input_nc, args.crop_size, args.crop_size, requires_grad=True).to(device)
    output_model = torch.load('./model/' + args.name + '.pt').to(device)
    torch_out = output_model(dummy_input1,dummy_input2)
    torch.onnx.export(model, (dummy_input1, dummy_input2), './onnx/'+ args.name +".onnx",
                      input_names = ['ChangeDetection_input1','ChangeDetection_input2'],
                      output_names = ['ChangeDetection_output'])
    
    '''
    #dynamic batch
    dummy_input1 = torch.randn(-1, args.input_nc, args.crop_size, args.crop_size, requires_grad=True).to(device)
    dummy_input2 = torch.randn(-1, args.input_nc, args.crop_size, args.crop_size, requires_grad=True).to(device)
    output_model = torch.load('./model/' + args.name + '.pt').to(device)
    torch_out = output_model(dummy_input1,dummy_input2)
    dynamic_axes = {'ChangeDetection_input1': {0: 'batch_size'}, 'ChangeDetection_input2': {0: 'batch_size'}, 'ChangeDetection_output': {0: 'batch_size'}}
    torch.onnx.export(model, (dummy_input1, dummy_input2), './onnx/'+ args.name +".onnx",
                      input_names = ['ChangeDetection_input1','ChangeDetection_input2'],
                      output_names = ['ChangeDetection_output'],
                      dynamic_axes=dynamic_axes)
    '''

    print('Onnx file is save in ./onnx/' + args.name + '.onnx')