# Quick Start Guide

Want to get this running fast? Here's the quick version - should take about 5 minutes!

## What You Need

- Python 3.8 or newer (check with `python --version`)
- About 10GB of free space (the model is big!)
- Internet connection (for downloading the model the first time)

## Installation (Super Simple)

### 1. Install Everything

**Easy way (if you're on Mac or Linux):**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual way (works everywhere):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fire It Up

```bash
streamlit run app.py
```

### 3. It Opens Automatically!

Your browser should pop open at `http://localhost:8501`. If not, just go there manually.

## Your First Image

1. Type something like: `"a beautiful sunset over mountains"`
2. Hit "Generate Images"
3. Wait a bit (30-60 seconds on CPU, 5-10 on GPU)
4. Boom! Download your creation!

## When Things Don't Work

**Model download taking forever?**
- Yeah, it's about 4GB. Go grab a coffee, it'll finish eventually.
- Make sure your internet is stable

**Getting out of memory errors?**
- Use smaller images (512x512)
- Generate just one at a time

**It's super slow?**
- Check if you're using GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Turn down inference steps to 20-30

## What's Next?

- Check out the full [README.md](README.md) for all the details
- Try different prompts - get creative!
- Play with negative prompts to avoid stuff you don't want
- Browse your gallery to see everything you've made

Have fun! 🎨

