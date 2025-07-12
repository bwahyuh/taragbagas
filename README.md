# **Intelligent Shoe Recommendation System with Multimodal AI**

Welcome to the repository for the Intelligent Shoe Recommendation System\! This is a sophisticated, end-to-end platform that leverages the power of multimodal Artificial Intelligence (AI) to provide users with accurate, context-aware, and highly relevant shoe product recommendations. Moving beyond the limitations of traditional keyword-based search, this application is designed to understand user queries through natural language, visual data (images), and a combination of both.

## **📜 Project Description**

The primary goal of this project is to address a common challenge in e-commerce: the gap between user intent and search results. Traditional systems often fail when users don't know the exact product name or use descriptive, subjective language. This project bridges that gap by integrating modern AI models to comprehend the semantic meaning behind text and the visual essence of an image, delivering a more intuitive and human-like shopping experience.

Its core feature is an intelligent chat assistant built on a **Retrieval-Aumented Generation (RAG)** architecture. This advanced design involves two key stages:

1. **Retrieval:** The system first retrieves a set of relevant product candidates from the database based on the user's query.  
2. **Generation:** These candidates are then passed as context to a powerful Large Language Model (LLM), which analyzes them and generates a coherent, well-reasoned, and helpful response.

This approach ensures that all recommendations are not only intelligent but are also factually grounded in the available product data, preventing the AI from "hallucinating" non-existent products.

## **✨ Key Features**

The application is equipped with a suite of search methods to provide a powerful and intuitive user experience:

1. **Keyword-Based Search:** The classic search functionality. Ideal for users who know exactly what they are looking for, the system performs a direct search for products whose titles contain the user-inputted keywords.  
2. **Semantic Search:** This feature unlocks the ability to search by meaning, not just words. Users can describe their needs in natural language (e.g., "comfortable shoes for summer travel" or "formal leather shoes for a wedding"), and the system uses vector similarity to find products that are semantically aligned with the query.  
3. **Image-Based Search:** Leveraging a powerful vision model, this allows users to upload an image of a shoe. The system then analyzes the visual features of the image and retrieves other products from the database that are visually similar in style, shape, and color.  
4. **Multimodal Search (Text \+ Image):** The most advanced search capability, combining the strengths of both text and image analysis. Users can upload an image and simultaneously add a descriptive text query (e.g., uploading a picture of running shoes and adding the text "in blue"). The system fuses these two modalities to perform a highly specific and targeted search.  
5. **Intelligent Chat Assistant (RAG):** This is the centerpiece of the application. It's an interactive chatbot that can understand complex, conversational questions. It performs the appropriate search method in the background, analyzes the retrieved candidates, and provides rich, detailed recommendations complete with justifications and key product features.

## **🛠️ Technology Stack**

This application is built on a modern technology stack designed for scalability, performance, and maintainability:

| Category | Technology | Purpose of Use |
| :---- | :---- | :---- |
| **Programming Language** | **Python** | The primary language for the entire backend, data processing, and AI model interaction. |
| **Application Framework** | **Streamlit** | Used to rapidly build an interactive and responsive user interface (UI), enabling real-time data visualization and user interaction. |
| **AI & ML Models** | **Vertex AI (Gemini)** | The brain behind the Intelligent Chat Assistant (RAG). Gemini is used to analyze user queries, understand the context from retrieved products, and generate intelligent, structured responses. |
|  | **Google SigLIP** | A multimodal AI model from Hugging Face that forms the core of the search capabilities. SigLIP is used to convert text and image inputs into vector representations (embeddings). |
|  | **Transformers & PyTorch** | Fundamental libraries used to load, run, and perform inference with the SigLIP model. |
| **Database** | **PostgreSQL** | The primary relational database used to store all product data, including titles, prices, descriptions, and features. |
|  | **pgvector** | A critical PostgreSQL extension that enables the efficient storage and querying of high-dimensional vector data, forming the basis of all semantic and image-based searches. |
| **Platform & Deployment** | **Google Cloud Platform (GCP)** | The primary cloud platform that serves as the foundation for the entire infrastructure. |
|  | **Cloud Run** | A serverless service for deploying the Streamlit application as a container, providing automatic scaling and simplified management. |
|  | **Artifact Registry** | Used as a private repository to store and manage the application's Docker images before deployment to Cloud Run. |
|  | **Cloud Build** | A CI/CD service that automates the process of building a Docker image from the source code and pushing it to the Artifact Registry. |

## **🧠 Key Concepts Explained**

This section details the core AI concepts that power this application.

#### **Recommendation System**

A Recommendation System is a type of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. In e-commerce, this translates to suggesting products to users. While basic systems rely on simple rules or keyword matching, this project implements an advanced system that uses **vector embeddings** to understand the *semantic context* of a user's query. This allows it to match "sepatu untuk lari di gunung" with a product titled "Men's Trail Running Shoes" even if the exact words don't match, because it understands the underlying meaning.

#### **Retrieval-Augmented Generation (RAG)**

RAG is an advanced AI architecture designed to make Large Language Models (LLMs) more accurate and reliable. It works in two main stages:

1. **Retrieval:** When a user asks a question, the system doesn't immediately ask the LLM. Instead, it first retrieves relevant information from a trusted data source (in this case, the PostgreSQL database). This is the "Retrieval" part.  
2. **Generation:** The retrieved information (the "context") is then bundled together with the original user question and sent to the LLM (Gemini). The LLM is instructed to generate an answer *only* based on the provided context. This is the "Augmented Generation" part.

The primary benefit of RAG is that it **grounds the LLM in facts**, preventing it from making up (or "hallucinating") information and ensuring the recommendations are based on actual, available products.

#### **Implementation in This Project**

This project masterfully combines these concepts in the **Intelligent Chat Assistant (Tab 5\)**:

* When you ask the chatbot a question, the **SigLIP model** acts as the **Retriever**. It converts your query (text or image) into a vector.  
* This vector is then used by **pgvector** to search the database and retrieve the top 10 most relevant product candidates.  
* These 10 products, along with their features, form the **Context**.  
* The Context and your original question are then passed to the **Gemini model**, which acts as the **Generator**.  
* Gemini follows a detailed prompt to analyze the candidates, select the best ones, and generate a detailed, well-reasoned response with bullet points explaining *why* each product is a good recommendation, based on its features.

This end-to-end RAG pipeline is what allows the chatbot to provide recommendations that are both intelligent and trustworthy.

## **🚀 How to Use the Application**

Each tab in the application is designed for a different use case. Here are some examples:

#### **1\. Keyword Search**

* **Objective:** To search for products with a known name.  
* **Example:** Enter "Piceno Soccer Shoe" in the search field to find products with that name.

#### **2\. Semantic Search**

* **Objective:** To find products based on a description of your needs.  
* **Example:** Write "formal leather shoes for a wedding" in the text area. The system will search for shoes that semantically match this description.

#### **3\. Image Search**

* **Objective:** To find shoes that are visually similar to an image you have.  
* **Example:** Upload an image of a red running shoe, and the system will display other running shoes with similar designs or colors.

#### **4\. Multimodal Search**

* **Objective:** To perform a highly specific search by combining visual and text inputs.  
* **Example:** Upload an image of a pair of boots, then add the text "that are waterproof and suitable for hiking". The system will look for boots that are visually similar AND meet the specified criteria.

#### **5\. Intelligent Chat Assistant**

* **Objective:** The most interactive and personalized shopping experience.  
* **Example:** You can ask directly in the chat input:  
  * "recommend me a pantofola d'oro shoe for playing soccer"  
  * "I need comfortable sandals for the beach"  
  * (While uploading an image) "are there any shoes like this but in black?"

The assistant will provide a detailed response, complete with the key features of each recommended product.

## **📄 License**

This project is protected under a proprietary license. Unauthorized copying, distribution, or modification of this code is strictly prohibited.

**Copyright © 2025 \- Bagas Wahyu Herdiansyah**

* **LinkedIn:** [bagas-wahyu-herdiansyah](https://www.linkedin.com/in/bagas-wahyu-herdiansyah/)  
* **GitHub:** [bwahyuh](https://github.com/bwahyuh)

All rights reserved.
