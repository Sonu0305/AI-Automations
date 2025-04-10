import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup, NavigableString
import re
import numpy as np

def setup_driver():
    """Sets up and returns a configured Chrome webdriver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_urls(base_url, num_pages=5):
    """Gets all the Pages URLs and saves them in a List"""
    url_list = [base_url]
    for page in range(2, num_pages + 1):
        next_url = base_url + f"&page={page}"
        url_list.append(next_url)
    return url_list

def extract_product_data(item):
    """Extract basic product data from search results page"""
    sponsored = False
    sponsored_tag = item.find('span', string=re.compile('Sponsored', re.IGNORECASE))
    if sponsored_tag:
        sponsored = True
    
    atag = item.h2.a if item.h2 else None
    if not atag:
        return None
    
    title = atag.text.strip()
    product_url = "https://www.amazon.in" + atag.get("href") if atag.get("href") else ""
    
    brand = ""
    brand_element = item.find('span', {'class': 'a-size-base-plus a-color-base'})
    if brand_element:
        brand = brand_element.text.strip()
    
    price = ""
    price_element = item.find('span', {'class': 'a-price-whole'})
    if price_element:
        price = price_element.text.strip().replace(",", "")
    
    rating = ""
    rating_element = item.find('span', {'class': 'a-icon-alt'})
    if rating_element:
        rating_text = rating_element.text
        rating_match = re.search(r'(\d+(\.\d+)?)', rating_text)
        if rating_match:
            rating = rating_match.group(1)
    
    reviews = ""
    reviews_element = item.find('span', {'class': 'a-size-base s-underline-text'})
    if reviews_element:
        reviews = reviews_element.text.strip().replace(",", "")
    
    img_url = ""
    img_element = item.find('img', {'class': 's-image'})
    if img_element:
        img_url = img_element.get('src', '')
    
    return {
        "Title": title,
        "Brand": brand,
        "Reviews": reviews,
        "Rating": rating,
        "Selling Price": price,
        "Image URL": img_url,
        "Product URL": product_url,
        "Sponsored": sponsored
    }

def scrape_amazon_products(search_term, num_pages=5):
    """Main scraping function"""
    base_url = f"https://www.amazon.in/s?k={search_term.replace(' ', '+')}&crid=2M096C61O4MLT&sprefix=ba%2Caps%2C283&ref=sr_pg_1"
    pages = get_urls(base_url, num_pages)
    
    driver = setup_driver()
    products_data = []
    
    try:
        for page_url in pages:
            print(f"Scraping: {page_url}")
            driver.get(page_url)
            time.sleep(2)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            results = soup.find_all('div', {'data-component-type': "s-search-result"})
            
            for item in results:
                product_data = extract_product_data(item)
                if product_data and product_data["Sponsored"]:
                    products_data.append(product_data)
            
            print(f"Found {len(products_data)} sponsored products so far...")
    
    finally:
        driver.quit()
    
    return products_data

def clean_data(df):
    """Clean and prepare the dataframe for analysis"""
    df = df.drop_duplicates(subset=['Title', 'Product URL'])
    
    df['Selling Price'] = pd.to_numeric(df['Selling Price'].str.replace('[^0-9]', '', regex=True), errors='coerce')
    
    df['Reviews'] = pd.to_numeric(df['Reviews'].str.replace('[^0-9]', '', regex=True), errors='coerce')
    
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    df['Brand'] = df.apply(lambda row: extract_brand_from_title(row), axis=1)
    
    df['Reviews'] = df['Reviews'].fillna(0)
    df['Rating'] = df['Rating'].fillna(0)
    
    return df

def extract_brand_from_title(row):
    """Extract brand from title if brand column is empty"""
    if pd.notna(row['Brand']) and row['Brand'].strip():
        return row['Brand']
    
    if pd.notna(row['Title']):
        first_word = row['Title'].split()[0]
        if len(first_word) > 1:
            return first_word
    
    return "Unknown"

def analyze_brand_performance(df):
    """Analyze brand performance"""
    brand_counts = df['Brand'].value_counts().reset_index()
    brand_counts.columns = ['Brand', 'Frequency']
    
    brand_ratings = df.groupby('Brand')['Rating'].mean().reset_index()
    brand_ratings.columns = ['Brand', 'Average Rating']
    
    brand_analysis = pd.merge(brand_counts, brand_ratings, on='Brand')
    brand_analysis = brand_analysis.sort_values('Frequency', ascending=False)
    
    return brand_analysis

def visualize_brand_performance(brand_analysis):
    """Create visualizations for brand performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    top_brands = brand_analysis.head(5)
    sns.barplot(x='Brand', y='Frequency', data=top_brands, ax=ax1, palette='viridis')
    ax1.set_title('Top 5 Brands by Frequency')
    ax1.set_xlabel('Brand')
    ax1.set_ylabel('Number of Products')
    ax1.tick_params(axis='x', rotation=45)
    
    top_brands_share = top_brands['Frequency'].sum() / brand_analysis['Frequency'].sum() * 100
    labels = list(top_brands['Brand']) + ['Others']
    sizes = list(top_brands['Frequency']) + [brand_analysis['Frequency'].sum() - top_brands['Frequency'].sum()]
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
    ax2.set_title('Market Share of Top Brands')
    
    plt.tight_layout()
    plt.savefig('brand_performance.png')
    plt.close()
    
    return 'brand_performance.png'

def analyze_price_vs_rating(df):
    """Analyze relationship between price and rating"""
    df['Rating Range'] = pd.cut(df['Rating'], 
                               bins=[0, 1, 2, 3, 4, 5], 
                               labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
    
    price_by_rating = df.groupby('Rating Range')['Selling Price'].mean().reset_index()
    
    df['Value Score'] = df['Rating'] / (df['Selling Price'] + 1)
    
    return df, price_by_rating

def visualize_price_vs_rating(df, price_by_rating):
    """Create visualizations for price vs rating analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(x='Rating', y='Selling Price', data=df, ax=ax1, alpha=0.7)
    ax1.set_title('Price vs. Rating Scatter Plot')
    ax1.set_xlabel('Rating (out of 5)')
    ax1.set_ylabel('Price (₹)')
    
    sns.barplot(x='Rating Range', y='Selling Price', data=price_by_rating, ax=ax2, palette='coolwarm')
    ax2.set_title('Average Price by Rating Range')
    ax2.set_xlabel('Rating Range')
    ax2.set_ylabel('Average Price (₹)')
    
    plt.tight_layout()
    plt.savefig('price_vs_rating.png')
    plt.close()
    
    return 'price_vs_rating.png'

def analyze_review_rating_distribution(df):
    """Analyze distribution of reviews and ratings"""
    top_by_reviews = df.sort_values('Reviews', ascending=False).head(5)
    
    top_by_rating = df[df['Reviews'] >= 10].sort_values('Rating', ascending=False).head(5)
    
    return top_by_reviews, top_by_rating

def visualize_review_rating_distribution(top_by_reviews, top_by_rating):
    """Create visualizations for review and rating distribution"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    sns.barplot(x='Reviews', y='Title', data=top_by_reviews, ax=ax1, palette='Blues_d')
    ax1.set_title('Top 5 Products by Number of Reviews')
    ax1.set_xlabel('Number of Reviews')
    ax1.set_ylabel('Product Title')
    
    sns.barplot(x='Rating', y='Title', data=top_by_rating, ax=ax2, palette='Greens_d')
    ax2.set_title('Top 5 Products by Rating (with at least 10 reviews)')
    ax2.set_xlabel('Rating (out of 5)')
    ax2.set_ylabel('Product Title')
    
    plt.tight_layout()
    plt.savefig('review_rating_distribution.png')
    plt.close()
    
    return 'review_rating_distribution.png'

def generate_insights(brand_analysis, df, price_by_rating, top_by_reviews, top_by_rating):
    """Generate actionable insights from the data"""
    insights = {
        "Brand Performance": {
            "Top Brand": brand_analysis.iloc[0]['Brand'],
            "Most Dominant Brands": list(brand_analysis.head(3)['Brand']),
            "Highest Rated Brand": brand_analysis.sort_values('Average Rating', ascending=False).iloc[0]['Brand'],
            "Untapped Potential": brand_analysis[brand_analysis['Frequency'] < 3].sort_values('Average Rating', ascending=False).head(3)['Brand'].tolist()
        },
        "Price vs. Rating": {
            "Price-Rating Correlation": df['Selling Price'].corr(df['Rating']),
            "Best Value Products": df.sort_values('Value Score', ascending=False).head(3)[['Title', 'Brand', 'Selling Price', 'Rating']].to_dict('records'),
            "Overpriced Products": df[(df['Selling Price'] > df['Selling Price'].quantile(0.75)) & 
                                   (df['Rating'] < df['Rating'].quantile(0.25))][['Title', 'Brand', 'Selling Price', 'Rating']].to_dict('records')
        },
        "Review & Rating": {
            "Most Reviewed Product": top_by_reviews.iloc[0]['Title'],
            "Highest Rated Popular Product": top_by_rating.iloc[0]['Title'],
            "Hidden Gems": df[(df['Reviews'] < df['Reviews'].quantile(0.5)) & 
                          (df['Rating'] > df['Rating'].quantile(0.75))].head(3)[['Title', 'Brand', 'Rating']].to_dict('records')
        }
    }
    
    return insights

def main():
    """Main function to execute the scraping and analysis workflow"""
    print("Starting scraping process...")
    products_data = scrape_amazon_products("soft toys", num_pages=3)
    
    raw_df = pd.DataFrame(products_data)
    raw_df.to_csv('raw_soft_toys_data.csv', index=False)
    print(f"Saved {len(raw_df)} raw products to CSV")
    
    print("Cleaning and preparing data...")
    clean_df = clean_data(raw_df)
    clean_df.to_csv('clean_soft_toys_data.csv', index=False)
    print(f"Saved {len(clean_df)} cleaned products to CSV")
    
    print("Analyzing data...")
    
    brand_analysis = analyze_brand_performance(clean_df)
    brand_viz_path = visualize_brand_performance(brand_analysis)
    print(f"Brand performance visualization saved to {brand_viz_path}")
    
    df_with_value, price_by_rating = analyze_price_vs_rating(clean_df)
    price_viz_path = visualize_price_vs_rating(df_with_value, price_by_rating)
    print(f"Price vs rating visualization saved to {price_viz_path}")
    
    top_by_reviews, top_by_rating = analyze_review_rating_distribution(clean_df)
    review_viz_path = visualize_review_rating_distribution(top_by_reviews, top_by_rating)
    print(f"Review and rating distribution visualization saved to {review_viz_path}")
    
    insights = generate_insights(brand_analysis, df_with_value, price_by_rating, top_by_reviews, top_by_rating)
    
    with open('amazon_insights.txt', 'w') as f:
        f.write("# Amazon Soft Toys Analysis Insights\n\n")
        
        f.write("## Brand Performance Insights\n")
        f.write(f"- Top Brand: {insights['Brand Performance']['Top Brand']}\n")
        f.write(f"- Most Dominant Brands: {', '.join(insights['Brand Performance']['Most Dominant Brands'])}\n")
        f.write(f"- Highest Rated Brand: {insights['Brand Performance']['Highest Rated Brand']}\n")
        f.write(f"- Untapped Potential Brands: {', '.join(insights['Brand Performance']['Untapped Potential'])}\n\n")
        
        f.write("## Price vs. Rating Insights\n")
        f.write(f"- Price-Rating Correlation: {insights['Price vs. Rating']['Price-Rating Correlation']:.2f}\n")
        f.write("- Best Value Products (High Rating, Low Price):\n")
        for product in insights['Price vs. Rating']['Best Value Products']:
            f.write(f"  * {product['Title']} by {product['Brand']} - ₹{product['Selling Price']:.0f} with {product['Rating']} stars\n")
        
        f.write("- Overpriced Products (High Price, Low Rating):\n")
        for product in insights['Price vs. Rating']['Overpriced Products']:
            f.write(f"  * {product['Title']} by {product['Brand']} - ₹{product['Selling Price']:.0f} with {product['Rating']} stars\n\n")
        
        f.write("## Review & Rating Insights\n")
        f.write(f"- Most Reviewed Product: {insights['Review & Rating']['Most Reviewed Product']}\n")
        f.write(f"- Highest Rated Popular Product: {insights['Review & Rating']['Highest Rated Popular Product']}\n")
        f.write("- Hidden Gems (High Rating, Few Reviews):\n")
        for product in insights['Review & Rating']['Hidden Gems']:
            f.write(f"  * {product['Title']} by {product['Brand']} - {product['Rating']} stars\n")
    
    print("Analysis complete! Insights saved to amazon_insights.txt")
    print("All visualizations saved as PNG files")

if __name__ == "__main__":
    main()