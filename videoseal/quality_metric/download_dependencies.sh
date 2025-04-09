
# MANIQA
git clone --depth=1 https://github.com/IIGROUP/MANIQA.git
cd MANIQA
    wget https://github.com/IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt -q --show-progress
    sed 's/score = torch\.tensor(\[\])\.cuda()/score = torch\.tensor(\[\])\.to(x\.device)/' -i models/maniqa.py
cd ..


# TReS
git clone --depth=1 https://github.com/isalirezag/TReS.git
cd TReS
    sed '/from openpyxl import load_workbook/d' -i folders.py
    sed 's/if float(torchvision.*/if False:/' -i misc.py
    sed 's/from resnet_modify  import resnet18 as resnet_modifyresnet/resnet_modifyresnet = resnet18/' -i models.py
    sed 's/from resnet_modify  import resnet34 as resnet_modifyresnet/resnet_modifyresnet = resnet34/' -i models.py
    sed 's/from resnet_modify  import resnet50 as resnet_modifyresnet/resnet_modifyresnet = resnet50/' -i models.py
    sed 's/from transformers import Transformer/from transformers2 import Transformer\nfrom resnet_modify import resnet18, resnet34, resnet50/' -i models.py
    mv transformers.py transformers2.py

    gdown 1yPfxyTj6q10Wkf6dxq6RAyuQSNOeJj7x -O bestmodel_1_2021-live.zip
    gdown 1kI-ZsEKSL1s6lT1XZnPA01PfCiwVAjGy -O bestmodel_1_2021-kadid10k.zip
    gdown 1OJAZzDCDtQct-4SbuQ4P_5-w3oCm0-Y5 -O bestmodel_1_2021-tid2013.zip
cd ..


# CONTRIQUE
git clone --depth=1 https://github.com/pavancm/CONTRIQUE.git
cd CONTRIQUE
    wget -L https://utexas.box.com/shared/static/rhpa8nkcfzpvdguo97n2d5dbn4qb03z8.tar -O models/CONTRIQUE_checkpoint25.tar -q --show-progress
cd ..


# AHIQ
git clone --depth 1 https://github.com/IIGROUP/AHIQ.git
cd AHIQ
    gdown https://drive.google.com/uc?id=1Nk-IpjnDNXbWacoh3T69wkSYhYYWip2W -O checkpoints/ahiq_pipal/
    # gdown https://drive.google.com/uc?id=1Jr2nLnhMA0f0uPEjMG7sH-T4WfIEasXn -O checkpoints/ahiq_pipal/
cd ..


# UVQ
git clone --depth 1 https://github.com/google/uvq.git
cd uvq/uvq_pytorch
    mv utils uvq_utils
    sed 's/from utils import custom_nn_layers/from uvq_utils import custom_nn_layers/' -i uvq_utils/compressionnet.py
    sed 's/from utils import custom_nn_layers/from uvq_utils import custom_nn_layers/' -i uvq_utils/contentnet.py
    sed 's/from utils import custom_nn_layers/from uvq_utils import custom_nn_layers/' -i uvq_utils/distortionnet.py
cd ../..
