import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("laptop_price - dataset.csv")


st.title('Data Visualization Using Streamlit')
st.markdown('`by Group 10 - BM7')


st.markdown("""    
The purpose of this is to convert our previous Data Visualization lesson in [Google Colab Notebook](https://colab.research.google.com/drive/151edx-zGRNvrjmuqXahFZ61FhGAxkGGC?usp=sharing) into a Streamlit Web Application.  
  
[GitHub Repository](https://github.com/princeVillamil/Group10-CSS145) for the source code
""")

# Follow this code #
# def bar_chart():
#   Code

# st.header("Bar Chart Demo") - Name of chart
# st.markdown("""

  # Ideal for comparing categorical data or displaying frequencies, bar charts offer a clear visual representation of values.  
    
  # **Observations:**  
  # 1. Notes
  # **Conclusion:**  
  # 1. Notes
# """)

# bar_chart()

def scatter_price_vs_weight():
  plt.figure(figsize=(10, 6))
  plt.scatter(df['Weight (kg)'], df['Price (Euro)'], color='blue', alpha=0.6, edgecolor='white',s=65)
  plt.title('Price Vs. Weight',fontsize=16, fontweight='bold')
  plt.xlabel('Weight (kg)')
  plt.ylabel('Price (Euro)')
  plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
  st.pyplot() 

st.header("Price Vs. Weight Scatter Plot")
scatter_price_vs_weight()
st.markdown("""    
  **Observations:**  
    This scatter plot will illustrate the correlation between a laptop's price and weight. From this graph, it can be observed that weight plays a significant role in its price. As you can see, as the weight of laptops increases, it will lead to higher prices. I can assume that heavier laptops contain additional features such as larger screens, larger batteries, and overall better hardware, which can significantly change the price.
""")
st.markdown("""    
  **Conclusion:**  
    In conclusion, being able to understand this correlation is crucial for consumers in order to find the best laptop in the market for their budget. On the other hand, if you were the manufacturer, it would also be possible to use this as an advantage for your laptops. For example, making larger laptops for a cheaper price may attract consumers.
""")



def gpu_distribution():
  plt.figure(figsize=(10, 9))
  plt.pie(df['GPU_Company'].value_counts().values, labels=df['GPU_Company'].value_counts().index, autopct='%0.2f%%', startangle=45, colors=['#00C7FD', '#76b900', '#ED1C24', '#0091BD'] , explode=[0.05, 0.05, 0.05, 0.05], wedgeprops={'edgecolor':'black', 'linewidth' : 0.5, 'antialiased' : True}, shadow=True)
  plt.axis('equal')
  plt.title('GPU Distribution', fontsize=16, fontweight='bold')
  st.pyplot() 

st.header("GPU Distribution Pie Chart")
gpu_distribution()
st.markdown("""    
  **Observations:**  
  1. This pie chart reveals which GPU brands are more prevalent from the date set provided to us, highlighting which brands hold a larger market share. As seen from the graph, Intel by far has a considerable amount of percentage of the market share compared to its competitors, with ARM being less than one percent.
""")
st.markdown("""    
  **Conclusion:**  
  1. This GPU distribution pie chart provides highlights into the market dynamics, showcasing which brands are currently leading in terms of GPUs, within the given dataset. This info can be vital for manufacturers and consumers because when making decision about product development or purchases it is possible to find the best GPU based on this chart.
""")

def Price_vs_CPUType(): 
  plt.figure(figsize=(10, 6))
  sns.boxplot(x='CPU_Type', y='Price (Euro)', data=df)
  plt.xticks(rotation=90)
  plt.title('Comparison of Laptop Price vs CPU_Type')
  plt.xlabel('CPU Type')
  plt.ylabel('Price (Euro)')
  plt.tight_layout()
  st.pyplot() 

def extract_resolution(resolution):
    parts = resolution.split()[-1]
    return parts
  
df['Resolution'] = df['ScreenResolution'].apply(extract_resolution)

def Price_vs_ScreenReso():
  plt.figure(figsize=(10, 6))
  sns.boxplot(x='Resolution', y='Price (Euro)', data=df)
  plt.xticks(rotation=90)
  plt.title('Comparison of Laptop Price vs Screen Resolution')
  plt.xlabel('Screen Resolution')
  plt.ylabel('Price (Euro)')
  plt.tight_layout()
  st.pyplot() 

st.header("Price Vs. CPU Type")
Price_vs_CPUType()
st.markdown("""    
  **Observations:**\n
  Comparing the Laptop price to the CPU Type, we can generally see that higher generation CPUs corresponds to much more expensive laptops. \n**Core i7** processors can be generally found within higher-priced laptops with its median price higher than other CPU types. \n**Core i5** processors have a much wider price range, which indicates that Core i5 CPUs are generally used within mid-range and higher-end priced laptops. \n**Core M** CPUs tend to be in the lower price range, likely due to their use in more lightweight and less powerful machines, such as thin ultrabooks. \n**Core i3** processors' price range are narrower, and they are found mostly in lower-priced models.
  """)

st.header("Price Vs. Screen Resolution")
Price_vs_ScreenReso()
st.markdown("""    
  **Observations:** \n
  Comparing the laptop Price to the Screen Resolution, we can generally see that the higher screen resolution a laptop has, the more expensive their prices get. Higher resolutions such as 2880x1800 and 2560x1600 are associated with more expensive laptops. This is consistent with the fact that high-end displays, often found in premium devices, tend to drive up the overall cost. 1920x1080 (Full HD) resolution laptops show a wide price range, suggesting this resolution is common across various price categories, from mid-range to premium. Lower resolutions such as 1366x768 are found in more affordable laptops, indicating that budget-friendly devices typically have less sharp displays.
  """)

st.header("Conclusion")
st.markdown("""    
  **Conclusion:** \n
  **CPU Type**: Laptops with more advanced CPUs (e.g., Core i7) tend to have higher prices. Budget laptops are more likely to feature lower generation CPUs. \n**Screen Resolution**: Higher screen resolutions are associated with more expensive laptops, suggesting that display quality is an important feature in premium devices. Full HD resolution is common across both mid-range and high-end laptops.
  """)



def scatter_price_vs_inches():
  plt.figure(figsize=(10, 6))
  plt.scatter(df['Inches'], df['Price (Euro)'], color='pink', alpha=0.5)
  plt.title('Laptop Price vs. Screen Size (Inches)')
  plt.xlabel('Screen Size (Inches)')
  plt.ylabel('Price (Euro)')
  plt.grid(True)
  st.pyplot() 


def scatter_price_vs_cpu_freq():
  plt.figure(figsize=(10, 6))
  plt.scatter(df['CPU_Frequency (GHz)'], df['Price (Euro)'], color='pink', alpha=0.5)
  plt.title('Laptop Price vs. CPU Frequency (GHz)')
  plt.xlabel('CPU Frequency (GHz)')
  plt.ylabel('Price (Euro)')
  plt.grid(True)
  st.pyplot() 
st.header("Price vs inches")
scatter_price_vs_inches()
st.markdown("""    
  **Observations:**  
  The graph "Laptop Price vs. Screen Size" shows no clear linear relationship between screen size and price, though there are patterns showcased in the data visualisation. For the 12 to 13-inch range, these laptops are priced below 2000 Euoros, with a few outliers. The laptops with 14 to 15-inch screens presented the widest range in pricing, from below 1000 Euros to above 4000 Euros. As for 16 to 18-inch laptops, they are generally priced higher, even exceeding 5000 Euros. Additionally, there were clusters, and prices of laptops that centred on certain price points such as 1000, 2000, and 3000 Euros.
  """)
st.markdown("""    
  **Conclusion:**  
  For the graph "Laptop Price vs. Screen Size", larger laptops (over 16 inches) tend to have higher price points compared to smaller ones. The highest-priced laptops are found in the 14 to 15-inch and 16 to 18-inch ranges, indicating that factors other than screen size play a significant role in determining price.
  """)

st.header("Price vs CPU Frequency")
scatter_price_vs_cpu_freq()
st.markdown("""    
  **Observations:**  
  The second graph titled, "Laptop Price vs CPU Frequency" presents the relationship between CPU frequency and determining prices for laptops. Most of the data points fall within the 1.0 GHz to 3.0 GHz, evidently clusters nearing the prices of 500 to 1000 Eeros can be seen in almost all CPU frequency levels. As for the mid-range, 2.0 to 2.5 GHz have a wider price range, extending from approximately 500 Euros to over 4000 Euros. Moreover, the higher-priced laptops (3000 Euros or above) tend to have CPU frequencies between 2.0 GHz to 3.0 GHz and have a significant price dispersion compared to lower frequencies.""")
st.markdown("""    
  **Conclusion:**  
  The graph "Laptop Price vs CPU Frequency" presents that higher CPU frequency generally correlates with higher prices, but the relationship is not strictly linear. As previously mentioned, this factor of the laptop does not determine the final price but contributes to it.""")




def scatter_price_vs_memory():
  plt.figure(figsize=(10, 6))
  plt.scatter(df['Memory'], df['Price (Euro)'], color='black')
  plt.title('Laptop Price vs. Memory')
  plt.xlabel('Memory')
  plt.ylabel('Price (Euro)')
  plt.xticks(rotation=90)
  plt.grid(True)
  st.pyplot() 

def scatter_price_vs_ram():
  plt.figure(figsize=(10, 6))
  plt.scatter(df['RAM (GB)'], df['Price (Euro)'], color='black')
  plt.title('Laptop Price vs. RAM')
  plt.xlabel('RAM (GB)')
  plt.ylabel('Price (Euro)')
  plt.grid(True)
  st.pyplot() 



st.header("Price vs Memory")
scatter_price_vs_memory()
st.markdown("""    
  **Observations:**  
    Viewing the graph, it is clear there is more of an aubdance of low memory devices, almost all HDDs under 500GB and SSDs under 250GB are under 3000 Euros. There is also a wide range of prices for the 1TB SSD ranging from 1900+ and the most expensive being over 6000 Euros. The prices cant fully accurately predict the storage on the PC as other components may also vary, but it is clear that laptops under 3000 mostly have 250GB SSD.
""")

st.markdown("""    
  **Conclusion:**  
    HDDs with 500GB and SSDs with 250GB a most common under the 3000 Euro mark, anything above this is is random as other components of the laptop are fluctuating the prices, The Memory size that has the most fluctuation is the 1TB SSD.
""")

st.header("Price vs CPU RAM")
scatter_price_vs_ram()
st.markdown("""    
  **Observations:**  
    There is a clear price to performance increase in terms of the RAM, with PCs under 3000 Euros having 16GB of RAM with less with a few exceptions. There are a few laptops in different price ranges that have 32GB RAM, ranging from 1000-6000 Euros.
""")

st.markdown("""    
  **Conclusion:**  
The cost of the PC in regards to the RAM is very straight-forward in a way that the RAM increases the more the laptop costs. The most common RAM sizes being 4, 8, 16 and 32. There are less and less laptops the higher the RAM is with 64GB only having 1 laptop. 32GB seems to fluctuate the most having a range of 1000 to 6000+
""")



def scatter_price_vs_gpucomp():
  plt.figure(figsize=(6, 10))
  plt.scatter(df['GPU_Company'], df['Price (Euro)'], color='red')
  plt.title('Price Vs. GPU Company')
  plt.xlabel('Weight (kg)')
  plt.ylabel('Price (Euro)')
  plt.grid(True)
  st.pyplot() 

def scatter_price_vs_RAM_by_company():
  plt.figure(figsize=(12, 10))

  # Get unique companies and plot
  companies = df['Company'].unique()
  colors = plt.cm.get_cmap('tab20', len(companies))

  for i, company in enumerate(companies):
      subset = df[df['Company'] == company]
      plt.scatter(subset['RAM (GB)'], subset['Price (Euro)'], color=colors(i), label=company, alpha=1, edgecolors='w', linewidth=0.5)

  # Add plot details
  plt.title('Price vs. RAM by Company')
  plt.xlabel('RAM (GB)')
  plt.ylabel('Price (Euro)')
  plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True)
  plt.tight_layout()

  # Show the plot
  st.pyplot() 

# call

st.header("Price vs GPU Company")
scatter_price_vs_gpucomp()
st.markdown("""    
  **OBSERVATIONS AND CONCLUSION**  
nvidia offers the most expensive GPUS, and AMD only offers cheap ones, with intel offering up to 3.1k euro gpus.

ASUS suprisingly offers 64 GB RAM for only 4k euro when Razer offers 32 GB for 6k, but generally, the more RAM, the more it costs.""")

st.header("Price vs CPU RAM by Company")
scatter_price_vs_RAM_by_company()
st.markdown("""    
  **OBSERVATIONS AND CONCLUSION**  
nvidia offers the most expensive GPUS, and AMD only offers cheap ones, with intel offering up to 3.1k euro gpus.

ASUS suprisingly offers 64 GB RAM for only 4k euro when Razer offers 32 GB for 6k, but generally, the more RAM, the more it costs.""")
