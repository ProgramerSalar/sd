import torch 
from torch import nn 


class LitEma(nn.Module):

    """ 
    Lightweight Exponential Moving Average (EMA) module for model parameters.
    Maintains shadow copies of model parameters that are updated with exponential decay.

    Args:
        model (nn.Module): The model whose parameters will be averaged 
        decay (float): Decay factor for EMA (typically close to 1, eg. 0.999)
        use_ema_updates (bool): Whether to use adaptive decay based on update count 
    """




    def __init__(self,
                 model,
                 decay=0.9999,
                 use_num_updates=True):
        
        super().__init__()
        # self.decay = decay
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1.")
        

        # Mapping from model parameter names to shadow buffer names 
        self.m_name2s_name = {}

        # Register decay as buffer (track number of updates if use_num_updates=True)
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates',
                             tensor=torch.tensor(0, dtype=torch.int) if use_num_updates else torch.tensor(-1, dtype=torch.int))
        

        # Initialize shadow parameters for all trainable parameters 
        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers 
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                # Register shadow parameter buffer initialized with current param value 
                self.register_buffer(s_name, p.clone().detach().data)

        # Storage for temporarily collected parameters 
        self.collected_params = []

    def forward(self, model):

        """ 
        Update shadow parameters using EMA.

        Args:
            model (nn.Module): The source model with current parameters
        """

        # Get base decay value 
        decay = self.decay 

        # Adaptive decay calculation (ramps up from 0 to self.decay during first updates)
        if self.num_updates >= 0:
            self.num_updates += 1 
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_min_decay = 1.0 - decay 

        # No gradient needed for EMA updates
        with torch.no_grad():
            # Get current model parameters and shadow parameters
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())


            for key in m_param:
                if m_param[key].requires_grad:

                    # Get corresponding shadow parameter name 
                    sname = self.m_name2s_name[key]

                    # Ensure type matching 
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])

                    # EMA update: shadow = shadow - (1- decay)*(shadow - current)
                    shadow_params[sname].sub_(one_min_decay * (shadow_params[sname] - m_param[key]))

                else:
                    # Verify we are not tracking non-trainable parameters
                    assert not key in self.m_name2s_name



            
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be temporarily stored.
        """

        self.collected_params = [param.clone() for param in parameters]



    def restore(self, parameters):

        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affacting the 
        original optimization process. Store the parameters before the 
        `copy_to` method. After validation (or model saving), use this to 
        restore the former parameters.

        Args:
            parameters: iterable of `torch.nn.Parameter`; the parameters to be 
                updated with the stored parameters.
        """

        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)



    def copy_to(self, model):

        """ 
        Copies the parameters from the shadow buffers (stored in this object)
        to the given model's parameters 

        This is typically used in techniques like Exponential Moving Average (EMA)
        where shadow parameters are maintained separately from the model's actual 
        parameters, and need to be periodically copied back to the model.

        Args:
            model (torch.nn.Module): the target model whose parameters will be 
                                    overwritten with the shadow parameters.
        """

        # Create a dictionary of the model's named parameters for easy access
        m_param = dict(model.named_parameters())

        # Create a dictionary of this object's named buffer (shadow parameters)
        shadow_params = dict(self.named_buffers())

        # Iterate through all parameters in the target model
        for key in m_param:
            # Check if parameter requires gradient (is trainable)
            if m_param[key].requires_grad:
                # Copy the corresponding shadow parameter's data to the model parameter
                # The shadow parameter is found using the name maping (m_name2s_name)
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)

            else:
                # For non-trainable parameters, verify they don't have shadow counterparts 
                # This is a safety check to ensure we are not missing any parameters
                assert not key in self.m_name2s_name




if __name__ == "__main__":

    

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
            self.fc = nn.Linear(64*28*28, 10)  # For classification to 10 classes
            
        def forward(self, x):
            x = self.conv1(x)
            x = x.view(x.size(0), -1)  # Flatten
            return self.fc(x)

    # Now this makes sense:
    model = MyModel()
    ema = LitEma(model)
    # print(ema)

    




