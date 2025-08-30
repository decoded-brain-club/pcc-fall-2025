# Kaggle Notebooks

The notebooks in this directory are meant to be imported directly to Kaggle and can be used without any changes to the code. The only change needed would be if you wanted to change the file path of something. 

## How to use Kaggle & Access GPU

1. Create a Kaggle account [here](kaggle.com)
2. Go to settings and verify your phone number
3. Refresh your browser and scroll down until you see **'Quotas'**. You should see something like this:
<img width="837" height="348" alt="Screenshot 2025-08-30 at 11 05 03 AM" src="https://github.com/user-attachments/assets/dcd0b0c9-2a48-4a2a-aa08-6fc583bbb835" />

4. On the left hand side, under the hamburger menu go to **'code'**
5. Click on **'New Notebook'** and you can either start from a new notebook or import an existing notebook. Under session options you should be able to see GPU and TPU options.
<img width="1604" height="926" alt="Screenshot 2025-08-30 at 11 08 24 AM" src="https://github.com/user-attachments/assets/12ebaea5-e8b8-496b-a79b-64d3b8b09305" />

6. Every time you switch from CPU, GPU and TPU the entire session restarts, meaning you have to run your code all over again so it's important to know when to run GPU and when not to

## How to upload a dataset to Kaggle

1. Go to datasets under the hamburger menu on the right hand side
2. Click on new dataset, and begin uploading
3. To use the dataset in your notebook go to your desired notebook
4. On the right hand side, under input, click **'add input'**
5. You can filter out by your own datasets by clicking **'Your Work'** and then press the add button. The result should look like the screenshot below

<img width="436" height="263" alt="Screenshot 2025-08-30 at 11 18 12 AM" src="https://github.com/user-attachments/assets/4efe7857-9860-47bf-a822-c9bbf95f333c" />
 
6. This will fetch the dataset from Kaggle's server into your own notebook (which acts as a sandbox). Meaning that your notebook still has to download the dataset again, so larger datasets can be very time consuming.
7. Once the dataset is added, you can run a cell for Kaggle to start loading the dataset into your notebook.
8. The file path to access the loaded dataset is **/kaggle/input/{dataset_name}**

