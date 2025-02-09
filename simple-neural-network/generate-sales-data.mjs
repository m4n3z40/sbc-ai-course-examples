import fs from 'fs';

function generateSalesData() {
    const data = [];
    let currentDate = new Date(2023, 0, 1); // 1º de Janeiro de 2023
    
    // Lista de feriados (simplificada)
    const holidays = {
        '1-1': 'Ano Novo',
        '4-21': 'Tiradentes',
        '5-1': 'Dia do Trabalho',
        '9-7': 'Independência',
        '10-12': 'Nossa Senhora',
        '11-2': 'Finados',
        '11-15': 'República',
        '12-25': 'Natal'
    };

    // Gerar dados para cada dia do ano
    for (let dayOfYear = 1; dayOfYear <= 365; dayOfYear++) {
        const month = currentDate.getMonth() + 1;
        const dayOfMonth = currentDate.getDate();
        const dayOfWeek = currentDate.getDay() + 1; // 1-7 (Domingo-Sábado)
        
        // Verificar se é feriado
        const isHoliday = holidays[`${month}-${dayOfMonth}`] ? 1 : 0;
        
        // Gerar faturamento base
        let revenue = 15000 + Math.random() * 5000; // Base entre 15000 e 20000
        
        // Ajustes por dia da semana
        if (dayOfWeek === 6 || dayOfWeek === 7) { // Fim de semana
            revenue *= 1.3; // 30% a mais
        }
        
        // Ajustes por feriado
        if (isHoliday) {
            revenue *= 1.5; // 50% a mais
        }
        
        // Ajustes sazonais
        if (month === 12) { // Dezembro
            revenue *= 1.4; // 40% a mais
        } else if (month === 1) { // Janeiro
            revenue *= 0.8; // 20% a menos
        }
        
        data.push({
            dayOfYear,
            dayOfMonth,
            dayOfWeek,
            month,
            isHoliday,
            revenue: Math.round(revenue * 100) / 100
        });
        
        currentDate.setDate(currentDate.getDate() + 1);
    }
    
    return data;
}

// Gerar CSV
const data = generateSalesData();
const header = 'dayOfYear,dayOfMonth,dayOfWeek,month,isHoliday,revenue\n';
const rows = data.map(row => 
    `${row.dayOfYear},${row.dayOfMonth},${row.dayOfWeek},${row.month},${row.isHoliday},${row.revenue}`
).join('\n');

fs.writeFileSync('./simple-neural-network/sales-data.csv', header + rows);
console.log('Arquivo sales-data.csv gerado com sucesso!'); 