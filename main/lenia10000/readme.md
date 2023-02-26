## What is Lenia10000

Lenia10000 - https://drive.google.com/drive/folders/1k20grcOSmCIFcDvrKJlaont33C6ALW2J?usp=share_link

Lenia10000 is a script for running combinations of mu ranging from 0.01 to 0.99 and sigma ranging from 0.01 to 0.99 while keeping following parameters fixed:

1. Growth Function: Gaussian Standard Function (From paper)
2. Growth Function's MU and Sigma values ranging from (0.01, 0.99)
3. Frames = 120
4. Board Initialisation = Random
5. Kernel Size = 16
6. Board_size = 64
7. OUTPUT_PATH = './new_outputs'
8. Kernel Base = spider_web_kernel(m=100,n=100) smoothing_factor = 0.5
```python 
def spider_web_kernel(self):
        # Following code is not using              
        m=100
        n=100
        # create a grid with zeros
        grid = np.zeros((n, m))
        
        # calculate the center of the grid
        center_x = n // 2
        center_y = m // 2
        
        # create a meshgrid
        x, y = np.meshgrid(np.arange(n), np.arange(m))
        
        # calculate the distance of each point from the center
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        
        # calculate the smoothing factor
        smoothing_factor = 0.5
        
        # calculate the values for each point
        grid = np.sin(distance * smoothing_factor) * distance
        
        return grid

```
9. Growth Function
```python 
def growth_function1(self, U:np.array):
        gaussian = lambda x, m, s: np.exp(-( (x-m)**2 / (2*s**2) ))
        return gaussian(U, self.mu, self.sigma)*2-1

```

10. How loop works

```python 
for i in range (100):
        for j in range(100):
            lenia = Lenia(mu=i/100, sigma=j/100)
            lenia.run_simulation(filename="mu{}_sigma{}".format(i/100,j/100))
```

Doing the math, we can understand that we have 100 values of mu from 0.01 to 0.99 and sigma 0.01 to 0.99 which makes a total of 100.100 = 10000 total combinations keeping rest of the configuration fixed. Remember we are talking about mu and sigma of Growth Function

Link for some initial outputs for mu ranges from [0.01 to 0.3] and sigma[0.01 to 0.99]


Some outputs

mu0.24 sigma0.18
!(https://drive.google.com/file/d/1PjM_3Qgpg6k7LLgrjar28S8jecafPnRb/view?usp=share_link)

