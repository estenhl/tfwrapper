
def create_experiment_history(name, **kwargs):
    """ Creates a jsonfile representing an entry of an experiment on 'name'_'timestamp'.json

        Parameters
        ----------
        name : str
            The name of the experiment. Will also be used for the filename
        

        Keyword parameters
        ------------------
        project : str
            Name of the project the experiment belongs to. Used for serverside ordering purposes
        description : str
            A brief description of the project
        model : object
            A description of the model used in the experiment. Requires the object to be stringifiable (uses the __str__ method)
        model_files : list
            A list of the model-files belonging to the experiment
        accuracy : float
            The accuracy achieved in the experiment
            
        

