import requests

def get_movie_matches(synopsis):
    url = "http://localhost:6000/find-match"
    payload = {"synopsis": synopsis}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()
        
        if not result:
            print("No matches found.")
            return

        print("\nMatched Movies:\n")
        for key, movie in result.items():
            print(f"Title: {movie['title']}")
            print(f"Year: {movie['released_year']}")
            print(f"Runtime: {movie['runtime']}")
            print(f"IMDb Rating: {movie['imdb_rating']}")
            print(f"Genres: {movie['genres']}")
            print(f"Poster: {movie['poster_url']}\n")
    
    except requests.exceptions.RequestException as e:
        print("Error:", e)


if __name__ == "__main__":
    print("Enter the movie synopsis or plot:")
    user_input = input("> ")
    get_movie_matches(user_input)
