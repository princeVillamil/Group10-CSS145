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
  1. This scatter plot will illustrate the correlation between a laptop's price and weight. From this graph, it can be observed that weight plays a significant role in its price. As you can see, as the weight of laptops increases, it will lead to higher prices. I can assume that heavier laptops contain additional features such as larger screens, larger batteries, and overall better hardware, which can significantly change the price.
""")
st.markdown("""    
  **Conclusion:**  
  1. In conclusion, being able to understand this correlation is crucial for consumers in order to find the best laptop in the market for their budget. On the other hand, if you were the manufacturer, it would also be possible to use this as an advantage for your laptops. For example, making larger laptops for a cheaper price may attract consumers.
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
