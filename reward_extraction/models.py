import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms

from r3m import load_r3m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utilities for defining neural nets
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, do_regularization=False):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        if do_regularization:
            mods = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        else:
            mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]

        for i in range(hidden_depth - 1):
            if do_regularization:
                mods += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            else:
                mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


# Define the forward model
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, output_mod=None, do_regularization=False):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, action_dim, hidden_depth, output_mod=output_mod, do_regularization=do_regularization)

    def forward(self, obs):
        next_pred = self.trunk(obs)
        return next_pred

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, output_mod=None, do_regularization=False):
        super().__init__()

        # critic should just output a single value!
        self.trunk = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth, output_mod=output_mod, do_regularization=do_regularization)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.trunk(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, h, w, hidden_fc=128, outputs=2, in_channels=3):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head1 = nn.Linear(linear_input_size, hidden_fc)
        self.head2 = nn.Linear(hidden_fc, outputs)
        self.h = h
        self.w = w
        self.hidden_fc = hidden_fc
        self.outputs = outputs

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # import ipdb; ipdb.set_trace()
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.head1(x))
        x = self.head2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(92416, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net = Net()

# Define the forward model
class ConvPolicy(nn.Module):
    def __init__(self, output_dim=2, coordconv=False):
        super().__init__()
        self.obs_trunk = ConvNet(h=84, w=84, hidden_fc=64, outputs=32)
        self.after_trunk = mlp(32 + 3, 128, output_dim, 1, output_mod=None)

    def forward(self, obs, goal):
        img_embedding = self.obs_trunk(obs)
        img_embedding = torch.cat([img_embedding, goal], dim=-1)
        next_pred = self.after_trunk(img_embedding)
        return next_pred


def get_resnet_preprocess_transforms():
    return transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
    ])

def get_unnormalize_transform():
    return transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ])

class ResnetPolicy(nn.Module):
    def __init__(
        self,
        output_size: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        do_preprocess: bool = True,
    ):
        super(ResnetPolicy, self).__init__()

        # get the backbone
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        fc_in_size = self.resnet.fc.in_features # 512
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        if freeze_backbone:
            assert pretrained
            for param in self.resnet.parameters():
                param.requires_grad = False

        # add our head to it
        self.output_size = output_size
        self.fc_head = nn.Sequential(
            nn.Linear(fc_in_size + 3, 256), # + 3 because of the goal
            nn.PReLU(),
            nn.Linear(256, output_size)
        )

        self.do_preprocess = do_preprocess
        self.preprocess_transforms = get_resnet_preprocess_transforms()

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        if self.do_preprocess:
            x = self.preprocess_transforms(x)
        x = self.resnet(x).squeeze(dim=2).squeeze(dim=2)
        x = torch.cat([x, goal], dim=1)
        x = self.fc_head(x)
        return x

    def save_model(self, model_path: str):
        print(f"saved frame classification model to {model_path}")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path: str):
        print(f"loaded frame classification model from {model_path}")
        self.load_state_dict(torch.load(model_path))

class R3MFeatureExtractor(nn.Module):
    def __init__(
        self,
        do_multiply_255: bool = True,
        freeze_backbone: bool = True,
    ):
        super(R3MFeatureExtractor, self).__init__()

        # get the backbone
        self.r3m = load_r3m("resnet18") # resnet18, resnet34, resnet50
        self.r3m.to(device)

        self.freeze_r3m = freeze_backbone
        if self.freeze_r3m:
            self.r3m.eval()
            for param in self.r3m.parameters():
                param.requires_grad = False

        self.r3m_embedding_dim = 512

        self.do_multiply_255 = do_multiply_255

    def forward(self, x: torch.Tensor):
        # r3m expects things to be [0-255] instead of [0-1]!!!
        if self.do_multiply_255:
            x = x * 255.0

        # some computational savings
        if self.freeze_r3m:
            with torch.no_grad():
                x = self.r3m(x)
        else:
            x = self.r3m(x)

        return x


class R3MPolicy(nn.Module):
    def __init__(
        self,
        output_size: int = 1,
        do_multiply_255: bool = True,
        freeze_backbone: bool = False,
        film_layer_goal: bool = False,
        state_only: bool = False
    ):
        super(R3MPolicy, self).__init__()
        self.state_only = state_only

        # get the backbone
        self.r3m = load_r3m("resnet18") # resnet18, resnet34, resnet50
        self.r3m.to(device)

        self.freeze_r3m = freeze_backbone
        if self.freeze_r3m:
            self.r3m.eval()
            for param in self.r3m.parameters():
                param.requires_grad = False

        self.r3m_embedding_dim = 512 # for resnet 18 - 512, for resnet50 - 2048

        # film layer the goal
        self.film_layer_goal = film_layer_goal
        self.goal_dim = 3
        if self.film_layer_goal:
            self.film_layer = nn.Sequential(
            nn.Linear(self.goal_dim, 4 * self.r3m_embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * self.r3m_embedding_dim, 2 * self.r3m_embedding_dim)
        )

        # add our head to the r3m output
        if self.film_layer_goal:
            fc_head_in = self.r3m_embedding_dim
        elif self.state_only:
            fc_head_in = self.r3m_embedding_dim
        else:
            fc_head_in = self.r3m_embedding_dim + 3
        self.output_size = output_size
        self.fc_head = nn.Sequential(
            nn.Linear(fc_head_in, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )

        self.do_multiply_255 = do_multiply_255

    def forward(self, x: torch.Tensor, goal: torch.Tensor = None):
        # r3m expects things to be [0-255] instead of [0-1]!!!
        if self.do_multiply_255:
            x = x * 255.0

        # some computational savings
        if self.freeze_r3m:
            with torch.no_grad():
                x = self.r3m(x)
        else:
            x = self.r3m(x)

        if not self.state_only:
            # film layer
            if self.film_layer_goal:
                gammabeta = self.film_layer(goal)
                gamma, beta = torch.split(gammabeta, self.r3m_embedding_dim, dim=1)
                x = x * gamma + beta
            else:
                # mix and run through head
                x = torch.cat([x, goal], dim=1)

        x = self.fc_head(x)
        return x

    def save_model(self, model_path: str):
        print(f"saved frame classification model to {model_path}")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path: str):
        print(f"loaded frame classification model from {model_path}")
        self.load_state_dict(torch.load(model_path))


class R3MImageGoalPolicy(nn.Module):
    def __init__(
        self,
        output_size: int = 1,
        do_multiply_255: bool = True,
        freeze_backbone: bool = False,
        film_layer_goal: bool = False, # kitchen goals are all the same, might make more sense with realworld videos
    ):
        super(R3MImageGoalPolicy, self).__init__()

        # get the backbone
        self.r3m = load_r3m("resnet18") # resnet18, resnet34, resnet50
        self.r3m.to(device)

        self.freeze_r3m = freeze_backbone
        if self.freeze_r3m:
            self.r3m.eval()
            for param in self.r3m.parameters():
                param.requires_grad = False

        self.r3m_embedding_dim = 512 # for resnet 18 - 512, for resnet50 - 2048

        # add our head to the r3m output
        fc_head_in = 2 * self.r3m_embedding_dim
        self.output_size = output_size
        self.fc_head = nn.Sequential(
            nn.Linear(fc_head_in, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )

        self.do_multiply_255 = do_multiply_255

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        # r3m expects things to be [0-255] instead of [0-1]!!!
        if self.do_multiply_255:
            x = x * 255.0
            goal = goal * 255.0

        # some computational savings
        if self.freeze_r3m:
            with torch.no_grad():
                x = self.r3m(x)
                goal = self.r3m(goal)
        else:
            x = self.r3m(x)
            goal = self.r3m(goal)

        # film layer
        x = torch.cat([x, goal], dim=1)
        x = self.fc_head(x)
        return x

    def save_model(self, model_path: str):
        print(f"saved frame classification model to {model_path}")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path: str):
        print(f"loaded frame classification model from {model_path}")
        self.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    test = R3MPolicy()
    import pdb; pdb.set_trace()