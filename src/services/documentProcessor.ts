import * as fs from 'fs';
import * as path from 'path';
import { PDFExtract } from 'pdf-extract';

export interface DocumentChunk {
  id: string;
  content: string;
  metadata: {
    source: string;
    page?: number;
    chunkIndex: number;
  };
}

export class DocumentProcessor {
  private chunkSize: number = 1000;
  private overlap: number = 200;

  async processDocument(filePath: string): Promise<DocumentChunk[]> {
    const extension = path.extname(filePath).toLowerCase();
    let content: string;

    switch (extension) {
      case '.pdf':
        content = await this.extractPDFText(filePath);
        break;
      case '.txt':
        content = fs.readFileSync(filePath, 'utf-8');
        break;
      default:
        throw new Error(`Formato de arquivo não suportado: ${extension}`);
    }

    return this.createChunks(content, filePath);
  }

  private async extractPDFText(filePath: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const pdfExtract = new PDFExtract();
      pdfExtract.extract(filePath, {}, (err: Error | null, data: { pages: { content: { str: string }[] }[] }) => {
        if (err) reject(err);
        else {
          const text = data.pages.map((page: { content: { str: string }[] }) => 
            page.content.map((item: { str: string }) => item.str).join(' ')
          ).join('\n');
          resolve(text);
        }
      });
    });
  }

  private createChunks(content: string, source: string): DocumentChunk[] {
    const chunks: DocumentChunk[] = [];
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    let currentChunk = '';
    let chunkIndex = 0;

    for (const sentence of sentences) {
      if (currentChunk.length + sentence.length > this.chunkSize) {
        if (currentChunk.trim()) {
          chunks.push({
            id: `${source}-${chunkIndex}`,
            content: currentChunk.trim(),
            metadata: {
              source,
              chunkIndex
            }
          });
          chunkIndex++;
        }
        
        // Mantém overlap entre chunks
        const words = currentChunk.split(' ');
        currentChunk = words.slice(-this.overlap / 10).join(' ') + ' ' + sentence;
      } else {
        currentChunk += sentence + '.';
      }
    }

    if (currentChunk.trim()) {
      chunks.push({
        id: `${source}-${chunkIndex}`,
        content: currentChunk.trim(),
        metadata: {
          source,
          chunkIndex
        }
      });
    }

    return chunks;
  }
}
