import streamlit as st




# Create the app title and header
st.title("Supplement Recommendation App")
st.subheader("Description:")
st.write("The AI Supplement Recommendation App is a user-friendly tool designed to assist individuals in finding the most suitable dietary supplements based on their personal preferences and requirements. By utilizing a Random Forest Classifier, the app analyzes a wide range of input factors, such as claimed improved aspects of fitness, fitness categories, and sport or exercise types, along with other metrics like evidence level and popularity, to determine the best-fitting supplements. The model used in this app has been trained on a dataset of supplements and their associated attributes, allowing it to make personalized supplement recommendations. The AI takes into consideration the latest research levels and the overall popularity of each supplement.")
st.subheader("How to Use:")
st.write("Users can access the app through the user interface, which provides a simple menu for input. Users specify their preferences by selecting claimed improved aspects of fitness, fitness categories, and sport or exercise types. Additionally, they can set preferences for evidence level and popularity. Once the user inputs these preferences, the app processes the data using the Random Forest Classifier to predict the top three supplements that align with the user's needs.")
st.subheader("Data Sources and Research References:")
st.write("- Cochrane Review: Jull et al (2008)	http://onlinelibrary.wiley.com/doi/10.1002/14651858.CD003892.pub3/abstract")
st.write("- Review: Pittler et al (2003)	http://onlinelibrary.wiley.com/o/cochrane/cldare/articles/DARE-12003000871/frame.html	")
st.write("- Steck et al (2007)	http://onlinelibrary.wiley.com/o/cochrane/clcentral/articles/134/CN-00588134/frame.html	")
st.write("- Colakoglu et al (2006)	http://onlinelibrary.wiley.com/o/cochrane/clcentral/articles/998/CN-00576998/frame.html	")
st.write("- Chen et al (2010)	http://onlinelibrary.wiley.com/o/cochrane/clcentral/articles/044/CN-00769044/frame.html	")
st.write("- Lorino et al (2006)	http://onlinelibrary.wiley.com/o/cochrane/clcentral/articles/725/CN-00574725/frame.html	")
st.write("- McLellan et al (2007)	http://onlinelibrary.wiley.com/o/cochrane/clcentral/articles/196/CN-00612196/frame.html")
st.write("- Backhouse et al (2011)	http://onlinelibrary.wiley.com/o/cochrane/clcentral/articles/805/CN-00799805/frame.html")
st.write("- Koh-Banerjee et al (2005)	http://onlinelibrary.wiley.com/o/cochrane/clcentral/articles/267/CN-00521267/frame.html	")
st.write("- Astorino et al (2010)	http://onlinelibrary.wiley.com/o/cochrane/clcentral/articles/696/CN-00786696/frame.html	")

