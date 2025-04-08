'use client'

import React, { useState } from 'react';

export default function DetectPage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setImageFile(e.target.files[0]);
      setOutputUrl(null); // Reset output
    }
  };

  const handleSubmit = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append('image', imageFile);

    setIsLoading(true);

    try {
      const res = await fetch("https://sem6-mini-project.onrender.com/detect", {
        method: 'POST',
        body: formData,
      });

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setOutputUrl(url);
    } catch (error) {
      console.error('Error uploading image:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-4 p-4">
      <h1 className="text-2xl font-bold">Upload Image for Detection</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button
        onClick={handleSubmit}
        disabled={!imageFile || isLoading}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
      >
        {isLoading ? 'Processing...' : 'Submit'}
      </button>

      {outputUrl && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">Detected Output:</h2>
          <img src={outputUrl} alt="Detection Result" className="max-w-md rounded shadow" />
        </div>
      )}
    </div>
  );
}
