import numpy as np
import pickle
import streamlit as st
import pandas as pd



# Sayfa Ayarları


st.set_page_config(
    page_title="Stroke Classifier",
    page_icon="data:image/jpeg;base64,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",
    menu_items={
        "Get help": "mailto:fatihxbayram@yandex.com.tr",
        "About": "For More Information\n" + "https://github.com/burakakay/Project-3-Machine-Learning-Classification-"
    }
)


# Başlık Ekleme
st.title("Stroke Classification Project")

# Markdown Oluşturma
st.markdown("This project aims to utilize machine learning for predicting the risk of stroke among hospital patients. The stroke prediction model evaluates the likelihood of stroke based on patients' demographic information and health characteristics.")
st.markdown("Explore the power of Machine Learning! click on the **:green[Stroke Prediction Demo]** tab in the left menu")
#Video Ekleme
# YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=-NJm4TJ2it0&ab_channel=TED-Ed"
# YouTube videosunu göm
st.video(youtube_url)
#İnme hakkında özet
st.write("Stroke is a life-threatening condition with severe consequences, including vision and speech loss, paralysis, and confusion. Those who have experienced a previous stroke face an elevated risk of further episodes. Annually, 15 million people worldwide suffer strokes, resulting in 5 million deaths and 5 million permanent disabilities. High blood pressure and tobacco use are major modifiable risk factors, with regulating blood pressure potentially preventing four out of ten stroke-related deaths in individuals under 65. While the incidence of stroke is declining in developed countries due to better control of risk factors, the aging population contributes to an overall increase in stroke cases.")


import streamlit as st

markdown_text = """

* ## Data Collection
The **Ministry of Health** has collected a dataset containing patient records and demographic information. Features in the dataset include **age**, **hypertension status**, history of **heart disease**, average **glucose level**, **body mass index (BMI)**, **gender**, **smoking status**, **marital status**, **work type**, and **residence type**.

* ## Machine Learning Model
The employed model involves a trained **classification algorithm** on the collected dataset. The model learns the interaction between features and the **stroke status** of patients, enabling it to predict the **stroke risk** for new patients.

* ## Application Interface
Patients can input the necessary information through a **web application**. The web application utilizes the **machine learning model** to assess the **stroke risk** based on the entered information.

* ## Results
The project can assist the hospital in predicting and proactively addressing the **stroke risk** among patients. Additionally, it contributes to informed health decision-making for patients.

---
**:red[Important Notice:]**
This project brief outlines a machine learning application designed to evaluate patients' health status. However, it's crucial to note that this application does not offer precise information about your health. For accurate health assessments, please consult with your primary healthcare provider. Your first doctor is the best source for personalized health guidance.

*We emphasize the importance of professional medical advice and encourage you to make informed decisions about your well-being under the guidance of qualified healthcare professionals.*


"""

# Streamlit Markdown
st.markdown(markdown_text)
st.markdown("## **Find the Nearest Hospital:**")



st.sidebar.success("Select a page above.")


import streamlit.components.v1 as components

# Load the HTML and JavaScript code
html_code = """
<!DOCTYPE html>
<html>

<head>
  <title>Map Example</title>
  <style>
    html,
    body,
    #map-canvas {
      height: 100%;
      margin: 0px;
      padding: 0px
    }
  </style>
  <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyDMPBP62NREvGoMfP6FT5vKKseKrNS60r8&v=3.exp&libraries=places"></script>
</head>

<body>
  <div id="map-canvas"></div>

  <script>
    var map;
    var services;
    var infowindow;

    function initialize() {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function (position) {
          var userLocation = new google.maps.LatLng(position.coords.latitude, position.coords.longitude);

          map = new google.maps.Map(document.getElementById('map-canvas'), {
            center: userLocation,
            zoom: 15
          });

          var request = {
            location: userLocation,
            radius: 200,
            types: ['hospital', 'health']
          };

          infowindow = new google.maps.InfoWindow();
          var service = new google.maps.places.PlacesService(map);
          service.nearbySearch(request, callback);
        }, function (error) {
          console.error("Error getting user location:", error);
        });
      } else {
        console.error("Geolocation is not supported by this browser.");
      }
    }

    function callback(results, status) {
      if (status == google.maps.places.PlacesServiceStatus.OK) {
        for (var i = 0; i < results.length; i++) {
          createMarker(results[i]);
        }
      }
    }

    function createMarker(place) {
      var placeLoc = place.geometry.location;
      var marker = new google.maps.Marker({
        map: map,
        position: place.geometry.location
      });

      google.maps.event.addListener(marker, 'click', function () {
        infowindow.setContent(place.name);
        infowindow.open(map, this);
      });
    }

    google.maps.event.addDomListener(window, 'load', initialize);
  </script>
</body>

</html>
"""

# Display the HTML and JavaScript code
components.html(html_code, height=600)



