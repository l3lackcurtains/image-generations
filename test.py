from generator import ImageGenerator

def main():
    # Create generator instance
    generator = ImageGenerator()
    
    # Print system info
    system_info = generator.get_system_info()
    print("\nSystem Information:")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    # Test image generation
    prompt = "A cat holding a sign that says hello world"
    print(f"\nGenerating image with prompt: {prompt}")
    
    image = generator.generate(
        prompt=prompt,
        height=512,
        width=512,
        model="flux-dev"
    )
    
    # Save the generated image
    image.save("flux-dev.png")
    print(f"Image saved as flux-dev.png using {generator.device}")

if __name__ == "__main__":
    main()
