from os import path
import random
import torch



def Gen_Data_id(FilePath='/home/ssd_2t/adl/code', TotalSamples=10375, Percentage=0.2):
    # test set
    Ts_index = random.sample(range(0, TotalSamples), int(TotalSamples * Percentage))
    filename = path.join(FilePath, "testID.txt")
    with open(filename, 'w') as file_write_obj:
        for var in Ts_index[0:int(TotalSamples * Percentage/2)]:
            file_write_obj.write(str(var))
            file_write_obj.write("\n")

    # # validate set
    # filename = path.join(FilePath, "validateID.txt")
    # with open(filename, 'w') as file_write_obj:
    #     for var in Ts_index[int(TotalSamples * Percentage/2):-1]:
    #         file_write_obj.write(str(var))
    #         file_write_obj.write("\n")

    # train set
    Tr_index = [x for x in range(0, TotalSamples) if x not in Ts_index]
    filename = path.join(FilePath, "trainID.txt")
    with open(filename, 'w') as file_write_obj:
        for var in Tr_index:
            file_write_obj.write(str(var))
            file_write_obj.write("\n")


def load_checkpoint(net, optimizer=None, checkpoint_path=None):
    if checkpoint_path:
        print('Load checkpoint: {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        if hasattr(net, 'module'):
            net.module.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == '__main__':
    Gen_Data_id()